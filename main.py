import asyncio
import logging
import os
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Union, Tuple
from fastapi import FastAPI, HTTPException, Depends, Header, File, UploadFile, Request, Form
from pydantic import BaseModel
import ray
from ray.util.state import list_nodes
import aiohttp
from aiohttp import ClientSession
import pandas as pd
import numpy as np
import scipy.stats as stats
import aiofiles
import re
from datetime import datetime
from fastapi.responses import StreamingResponse
import json
from prometheus_fastapi_instrumentator import Instrumentator
from contextlib import asynccontextmanager
import shutil
import hashlib
from logging.handlers import RotatingFileHandler
import jieba
import jieba.posseg as pseg
import uuid
import fasttext
from cachetools import TTLCache
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
import magic
import mimetypes

# Configure structured JSON logging
log_formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
)
file_handler = RotatingFileHandler(
    "/app/fastapi.log", mode="a", maxBytes=10*1024*1024, backupCount=2
)
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logging.basicConfig(
    level=logging.DEBUG,
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

# Initialize jieba without custom dictionary
jieba.initialize()

# Initialize SentenceTransformer for query rewriting
try:
    embedder = SentenceTransformer("/models/BAAI/bge-large-zh-v1.5")
    logger.info("Loaded SentenceTransformer model: bge-large-zh-v1.5")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    embedder = None

# Query rewrite templates
QUERY_TEMPLATES = [
    "简述 {关键词} 的主要内容",
    "分析 {关键词} 的关键点",
    "总结 {关键词} 的核心信息",
    "描述 {关键词} 的特点",
    "查询 {关键词} 的相关数据"
]

# FastAPI application
app = FastAPI()

# Model service configuration
VLLM_URL = "http://192.168.0.125:8000/v1/completions"
VLLM_MODEL = os.getenv("VLLM_MODEL", "/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
OLLAMA_URL = "http://192.168.0.126:11434/api/generate"
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 1200
GEMMA_TIMEOUT_SECONDS = 600
NUM_SAMPLES = 1
BATCH_SIZE = 10
MAX_CONTEXT_CHARS = 8000
STREAMING_BUFFER_SIZE = 1024

# Ray initialization state
ray_initialized = False
vector_store_actors = []

# Cache and data directories
cache_dir = "/mnt/nfs_doc_cache"
container_data_dir = "/mnt/nfs_doc_cache"

# In-memory caches
vector_cache = TTLCache(maxsize=1000, ttl=300)
response_cache = TTLCache(maxsize=500, ttl=600)

# Request models
class QueryRequest(BaseModel):
    query: str
    task_type: str = "search"
    file_names: Optional[List[str]] = None
    max_results: Optional[int] = 5
    output_format: Optional[str] = "plain_text"

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict

# API key validation
async def verify_api_key(x_api_key: str = Header(default=None)):
    expected_key = os.getenv("API_KEY", "your-secret-key")
    if x_api_key != expected_key:
        logger.error(f"Invalid X-API-Key: received '{x_api_key}', expected '{expected_key}'")
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

async def verify_openai_api_key(request: Request):
    expected_key = os.getenv("API_KEY", "your-secret-key")
    x_api_key = request.headers.get("X-API-Key")
    authorization = request.headers.get("Authorization")

    if x_api_key == expected_key:
        return x_api_key
    elif authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:].strip()
        if api_key == expected_key:
            return api_key
        else:
            logger.error(f"Invalid Bearer token: received '{api_key}', expected '{expected_key}'")
            raise HTTPException(status_code=401, detail="Invalid API key")
    else:
        logger.error(f"Missing or invalid headers: X-API-Key='{x_api_key}', Authorization='{authorization}'")
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

# Health endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Root endpoint
@app.get("/")
async def root():
    return {"message": "FastAPI server is running"}

# Models endpoint
@app.get("/v1/models", dependencies=[Depends(verify_openai_api_key)])
async def list_models(request: Request):
    headers = dict(request.headers)
    logger.info(f"Received /v1/models request: headers={headers}")
    return {
        "object": "list",
        "data": [
            {
                "id": "fastapi-multi-model",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "fastapi"
            }
        ]
    }

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

# Handle SIGTERM
def handle_sigterm(*args):
    logger.info("Received SIGTERM, initiating graceful shutdown")
    loop = asyncio.get_event_loop()
    loop.stop()
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()
    logger.info("Event loop closed, exiting")
    os._exit(15)

if threading.current_thread() is threading.main_thread():
    signal.signal(signal.SIGTERM, handle_sigterm)
else:
    logger.warning("SIGTERM handler not set: not in main thread")

# HTTP client session with connection pooling
async def get_aiohttp_session() -> ClientSession:
    try:
        connector = aiohttp.TCPConnector(limit=100)
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT_SECONDS),
            connector=connector
        )
    except Exception as e:
        logger.error(f"Failed to create aiohttp session: {str(e)}")
        raise HTTPException(status_code=500, detail="Session creation failed")

# Initialize Ray with retry
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def init_ray():
    try:
        ray.init(address="192.168.0.125:6379", ignore_reinit_error=True)
        nodes = list_nodes()
        if not nodes:
            logger.warning("No Ray nodes found, proceeding without Ray.")
            return False
        logger.info(f"Connected to Ray cluster with {len(nodes)} nodes")
        return True
    except Exception as e:
        logger.error(f"Ray initialization failed: {e}")
        raise

# Actor caching utilities
def save_actor_cache(actor_names: List[str]):
    """Save the list of healthy actor names to a file."""
    cache_path = os.path.join(cache_dir, "actors.json")
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"actors": actor_names}, f)
        logger.info(f"Saved actor cache to {cache_path}: {actor_names}")
    except Exception as e:
        logger.error(f"Failed to save actor cache to {cache_path}: {e}")

def load_actor_cache() -> List[str]:
    """Load the list of healthy actor names from a file."""
    cache_path = os.path.join(cache_dir, "actors.json")
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                actors = data.get("actors", [])
                logger.info(f"Loaded actor cache from {cache_path}: {actors}")
                return actors
        logger.info(f"No actor cache found at {cache_path}")
        return []
    except Exception as e:
        logger.error(f"Failed to load actor cache from {cache_path}: {e}")
        return []

# Initialize Ray and fetch VectorStore actors
@asynccontextmanager
async def lifespan(app: FastAPI):
    global ray_initialized, vector_store_actors
    try:
        if not ray_initialized:
            ray_initialized = await init_ray()
            if ray_initialized:
                max_actors = 3
                vector_store_actors = []
                # Try loading from cache first
                cached_actor_names = load_actor_cache()
                valid_cached_actors = []
                if cached_actor_names:
                    logger.info(f"Validating {len(cached_actor_names)} cached actors")
                    for actor_name in cached_actor_names:
                        try:
                            actor = ray.get_actor(actor_name, namespace="vector_store")
                            start_time = time.time()
                            health = ray.get(actor.health_check.remote(), timeout=120)
                            duration = time.time() - start_time
                            if (health.get("status") == "healthy" and 
                                health.get("base_index_size", -1) >= 0):
                                logger.info(f"Validated cached actor {actor_name} in {duration:.2f}s: {health}")
                                valid_cached_actors.append(actor)
                            else:
                                logger.warning(f"Cached actor {actor_name} unhealthy: {health}, killing actor")
                                ray.kill(actor)
                        except Exception as e:
                            logger.warning(f"Failed to validate cached actor {actor_name}: {e}")
                    vector_store_actors.extend(valid_cached_actors)
                
                # If we don't have enough valid actors, discover or create new ones
                if len(vector_store_actors) < max_actors:
                    logger.info(f"Need {max_actors - len(vector_store_actors)} more actors, discovering new ones")
                    for i in range(max_actors):
                        actor_name = f"vector_store_{i}"
                        if any(actor_name in str(actor) for actor in vector_store_actors):
                            logger.info(f"Actor {actor_name} already included, skipping")
                            continue
                        try:
                            actor = ray.get_actor(actor_name, namespace="vector_store")
                            start_time = time.time()
                            health = ray.get(actor.health_check.remote(), timeout=120)
                            duration = time.time() - start_time
                            if (health.get("status") == "healthy" and 
                                health.get("base_index_size", -1) >= 0):
                                logger.info(f"Found healthy actor {actor_name} in {duration:.2f}s: {health}")
                                vector_store_actors.append(actor)
                            else:
                                logger.warning(f"Actor {actor_name} unhealthy: {health}, killing actor")
                                ray.kill(actor)
                        except ValueError:
                            logger.info(f"Actor {actor_name} not found, skipping creation as enterprise_rag.py handles it")
                        except Exception as e:
                            logger.warning(f"Failed to check actor {actor_name}: {e}")
                
                # Ensure exactly 3 actors
                if len(vector_store_actors) != max_actors:
                    logger.error(f"Expected {max_actors} actors, but found {len(vector_store_actors)}")
                    log_path = "/app/vector_store.log"
                    if os.path.exists(log_path):
                        with open(log_path, "r") as f:
                            log_content = f.read()[:1000]
                            logger.error(f"vector_store.log content: {log_content}")
                    raise RuntimeError(f"Failed to initialize {max_actors} VectorStore actors")
                
                # Save the validated actor list to cache
                actor_names = [f"vector_store_{i}" for i in range(max_actors)]
                save_actor_cache(actor_names)
                logger.info(f"Connected to {len(vector_store_actors)} VectorStore actor(s): {actor_names}")
        yield
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        if ray_initialized:
            ray.shutdown()
            ray_initialized = False
        logger.info("Ray shutdown complete")

app.router.lifespan_context = lifespan

# Utility functions
def clean_query(query: str, custom_prompt: Optional[str]) -> Tuple[str, Optional[str]]:
    cleaned_query = query.replace("@#%", "").strip()
    cleaned_query = re.sub(r'\*=\d+', '', cleaned_query).strip()
    cleaned_query = re.sub(r'^\.+', '', cleaned_query).strip()
    extracted_prompt = custom_prompt

    if extracted_prompt and extracted_prompt in cleaned_query:
        cleaned_query = cleaned_query.replace(extracted_prompt + "。", "").replace(extracted_prompt, "").strip()
        logger.debug(f"Removed custom prompt from query: {cleaned_query}")

    if not extracted_prompt:
        if "*&" in cleaned_query:
            start_idx = cleaned_query.index("*&") + 2
            end_idx = cleaned_query.find("。", start_idx)
            if end_idx != -1:
                extracted_prompt = cleaned_query[start_idx:end_idx].strip()
                cleaned_query = (cleaned_query[:cleaned_query.index("*&")] + cleaned_query[end_idx+1:]).strip()
            else:
                extracted_prompt = cleaned_query[start_idx:].strip()
                cleaned_query = cleaned_query[:cleaned_query.index("*&")].strip()
        elif "提示词：" in cleaned_query:
            start_idx = cleaned_query.index("提示词：") + len("提示词：")
            end_idx = cleaned_query.find("。", start_idx)
            if end_idx != -1:
                extracted_prompt = cleaned_query[start_idx:end_idx].strip()
                cleaned_query = (cleaned_query[:cleaned_query.index("提示词：")] + cleaned_query[end_idx+1:]).strip()
            else:
                extracted_prompt = cleaned_query[start_idx:].strip()
                cleaned_query = cleaned_query[:cleaned_query.index("提示词：")].strip()

    cleaned_query = re.sub(r'\*&\s*', '', cleaned_query).strip()
    cleaned_query = re.sub(r'\。\s*', '', cleaned_query).strip()
    logger.debug(f"Final cleaned query for vector store: {cleaned_query}")
    return cleaned_query.strip(), extracted_prompt

def extract_custom_prompt(query: str) -> Optional[str]:
    match = None
    if "提示词：" in query:
        start_idx = query.index("提示词：") + len("提示词：")
        match = query[start_idx:]
    elif "*&" in query:
        start_idx = query.index("*&") + 2
        match = query[start_idx:]
    if match:
        end_idx = match.find("。")
        if end_idx != -1:
            return match[:end_idx].strip()
        else:
            logger.warning("No '。' found after custom prompt marker, using remaining query as prompt")
            return match.strip()
    return None

def extract_timeout(query: str) -> int:
    match = re.search(r'\*=\s*(\d+)', query)
    if match:
        try:
            timeout = int(match.group(1))
            if timeout <= 0:
                logger.warning(f"Invalid timeout value {timeout}, using default {DEFAULT_TIMEOUT_SECONDS}")
                return DEFAULT_TIMEOUT_SECONDS
            if timeout > MAX_TIMEOUT_SECONDS:
                logger.warning(f"Timeout {timeout} exceeds maximum {MAX_TIMEOUT_SECONDS}, capping at {MAX_TIMEOUT_SECONDS}")
                return MAX_TIMEOUT_SECONDS
            return timeout
        except ValueError:
            logger.warning(f"Failed to parse timeout from query: {query}, using default {DEFAULT_TIMEOUT_SECONDS}")
            return DEFAULT_TIMEOUT_SECONDS
    return DEFAULT_TIMEOUT_SECONDS

def assign_actor_for_file(file_name: str, actors: List) -> int:
    if not actors:
        logger.warning("No actors available, returning index 0")
        return 0
    hash_value = int(hashlib.md5(file_name.encode()).hexdigest(), 16)
    actor_idx = hash_value % len(actors)
    logger.debug(f"Assigned file {file_name} to actor index {actor_idx}")
    return actor_idx

async def query_model(session: ClientSession, url: str, payload: Dict, retries: int = 5, num_samples: int = 1, timeout: int = 300) -> List[Optional[str]]:
    responses = []
    for i in range(num_samples):
        start_time = time.time()
        payload["request_id"] = str(uuid.uuid4())
        logger.info(f"Sample {i+1}/{num_samples} - Sending POST to {url} with payload: {json.dumps(payload, ensure_ascii=False)}")
        for attempt in range(retries):
            try:
                async with session.post(url, json=payload, timeout=timeout) as response:
                    duration = time.time() - start_time
                    logger.info(f"Sample {i+1}/{num_samples} - Received response from {url}: status={response.status}, duration={duration:.2f}s")
                    if response.status != 200:
                        logger.warning(f"Request to {url} failed with status {response.status}")
                        responses.append(None)
                        break
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' not in content_type:
                        logger.warning(f"Unexpected content type from {url}: {content_type}")
                        responses.append(None)
                        break
                    text = await response.text()
                    if '<html' in text.lower():
                        logger.warning(f"Response from {url} contains HTML content: {text[:200]}...")
                        responses.append(None)
                        break
                    try:
                        data = json.loads(text)
                        if "choices" in data:
                            text = data.get("choices", [{}])[0].get("text", "")
                        elif "response" in data:
                            text = data.get("response", "")
                        if text and len(text) > 10:
                            text = text.replace("<think>", "").replace("</think>", "").strip()
                            text = re.sub(r'```json\n|```', '', text).strip()
                            text = re.sub(r'[\x00-\x1F\x7F]', '', text)
                            if "generate_title" not in payload.get("prompt", ""):
                                responses.append(text)
                                break
                            try:
                                parsed = json.loads(text)
                                if isinstance(parsed, (dict, list)):
                                    text = json.dumps(parsed)
                                    responses.append(text)
                                else:
                                    logger.warning(f"Invalid JSON structure: {text}")
                                    responses.append(None)
                            except json.JSONDecodeError:
                                json_match = re.search(r'\{[\s]*"[^"]+"[\s]*:[\s]*(?:\[([^\]]*)\]|".*?")[\s]*\}', text)
                                if json_match:
                                    tags_str = json_match.group(0)
                                    try:
                                        parsed = json.loads(tags_str)
                                        text = json.dumps(parsed)
                                        responses.append(text)
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"Failed to parse JSON from response: {text}, error: {e}")
                                        responses.append(None)
                                else:
                                    logger.warning(f"Invalid JSON format in response: {text}")
                                    responses.append(None)
                        else:
                            logger.warning(f"Invalid response: {text}")
                            responses.append(None)
                        break
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse response as JSON: {text[:200]}..., error: {e}")
                        if "generate_title" not in payload.get("prompt", ""):
                            responses.append(text)
                        else:
                            responses.append(None)
                        break
            except asyncio.TimeoutError:
                logger.error(f"Timeout after {timeout}s while querying {url}")
                responses.append(None)
                break
            except aiohttp.ClientError as e:
                logger.warning(f"Attempt {attempt + 1}/{retries} to {url} failed: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"All retries failed for {url}: {str(e)}")
                    responses.append(None)
                await asyncio.sleep(2 ** attempt)
    return responses

def compute_context_adherence(response: str, context: str, keywords: List[str]) -> float:
    if not response or not context:
        return 0.0
    try:
        response_lower = response.lower()
        context_lower = context.lower()
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in response_lower)
        context_overlap = sum(1 for word in response_lower.split() if word in context_lower)
        
        max_possible_matches = max(len(keywords), 1)
        max_possible_overlap = max(len(response_lower.split()), 1)
        keyword_score = keyword_matches / max_possible_matches
        overlap_score = context_overlap / max_possible_overlap
        
        final_score = 0.7 * keyword_score + 0.3 * overlap_score
        logger.debug(f"Context adherence - Keywords: {keyword_matches}/{max_possible_matches}, Overlap: {context_overlap}/{max_possible_overlap}, Score: {final_score:.2f}")
        return min(final_score, 1.0)
    except Exception as e:
        logger.error(f"Context adherence error: {str(e)}. Returning default score.")
        return 0.0

def compute_cross_model_scores(model_responses: Dict[str, str]) -> Dict[str, float]:
    if not model_responses or len(model_responses) < 2:
        return {model: 0.0 for model in model_responses}
    try:
        responses = [r for r in model_responses.values() if r]
        models = [m for m, r in model_responses.items() if r]
        if len(responses) < 2:
            return {model: 0.0 for model in model_responses}
        from difflib import SequenceMatcher
        scores = {}
        for i, model in enumerate(models):
            sim_scores = [
                SequenceMatcher(None, responses[i], responses[j]).ratio()
                for j in range(len(responses)) if j != i
            ]
            scores[model] = sum(sim_scores) / len(sim_scores) if sim_scores else 0.0
        max_score = max(scores.values(), default=1.0)
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        return scores
    except Exception as e:
        logger.error(f"Cross-model scoring error: {str(e)}. Returning default scores.")
        return {model: 1.799 / len(model_responses) for model in model_responses}

def select_best_response(samples: List[Optional[str]], context_score: float, context: str, keywords: List[str]) -> tuple[Optional[str], float]:
    if not samples or not any(samples):
        return None, context_score
    valid_samples = [s for s in samples if s]
    if len(valid_samples) == 1:
        return valid_samples[0], context_score
    if context_score < 0.5:
        return max(valid_samples, key=len), context_score
    return valid_samples[0], context_score

def extract_file_name(query: str) -> Optional[str]:
    match = re.search(r'[\w-]+\.(?:txt|csv|xlsx|xls|json|pdf|docx|jpg|jpeg|png|ppt|pptx)', query, re.IGNORECASE)
    return match.group(0) if match else None

def extract_keywords(query: str) -> List[str]:
    query = query.get("content", query) if isinstance(query, dict) else query
    query = query.lower()
    file_name = extract_file_name(query)
    if file_name:
        query = query.replace(file_name, "")
    query = re.sub(r'### Task:.*?### Chat History:.*?(USER:.*?)(?:ASSISTANT:.*)?$', r'\1', query, flags=re.DOTALL)
    query = re.sub(r'summarize|content|of|in|approximately|100|words|compare|analyze|抱歉|没有|找到|提供|相关|名为', '', query)
    words = pseg.cut(query)
    stopwords = {'使用', '查询', '回复', '请', '和', '在', '之后', '是', '的', '了', '所有', '文件', '记录', '列出'}
    keywords = [word for word, flag in words if flag in ['n', 'v', 'nr'] and len(word) > 1 and word not in stopwords]
    seen = set()
    keywords = [word for word in keywords if not (word in seen or seen.add(word))]
    logger.info(f"Extracted keywords: {keywords}")
    return keywords

async def hybrid_search(query: str, file_names: Optional[List[str]], max_results: int) -> List[Dict]:
    cache_key = f"{query}:{':'.join(file_names or [])}:{max_results}"
    if cache_key in vector_cache:
        logger.info(f"Cache hit for query: {query}")
        return vector_cache[cache_key]

    search_results = []
    tasks = []
    for actor in vector_store_actors:
        tasks.append(
            asyncio.get_event_loop().run_in_executor(
                None,
                lambda a=actor: ray.get(
                    a.search_across_files.remote(
                        query=query,
                        k=max_results * 2,
                        file_names=file_names
                    ),
                    timeout=60
                )
            )
        )
    try:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, list):
                search_results.extend(result)
            else:
                logger.warning(f"Search failed for an actor: {str(result)}")
    except Exception as e:
        logger.warning(f"Search failed: {e}")

    if not search_results:
        logger.warning(f"No search results for query: {query}, file_names: {file_names}")
        vector_cache[cache_key] = []
        return []

    keywords = extract_keywords(query)
    seen_chunks = set()
    unique_results = []
    for result in sorted(search_results, key=lambda x: x.get("distance", float("inf"))):
        chunk_id = result.get("chunk_id")
        if chunk_id and chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            content = result.get("content", "").lower()
            match_count = sum(1 for keyword in keywords if keyword.lower() in content)
            distance = result.get("distance", float("inf"))
            keyword_score = match_count / max(len(keywords), 1) if keywords else 0.5
            similarity_score = 1.0 - min(distance, 1.0)
            result["relevance_score"] = 0.7 * similarity_score + 0.3 * keyword_score
            unique_results.append(result)

    final_results = sorted(unique_results, key=lambda x: x.get("relevance_score", 0.0), reverse=True)[:max_results]
    if not final_results:
        logger.warning(f"No relevant results after filtering for query: {query}")

    vector_cache[cache_key] = final_results
    logger.info(f"Cached {len(final_results)} search results")
    return final_results

async def rewrite_query(query: str, session: ClientSession) -> str:
    if not embedder:
        logger.warning("SentenceTransformer not available, returning original query")
        return query
    try:
        # Extract keywords for template filling
        keywords = extract_keywords(query)
        keyword_str = " ".join(keywords)

        # Generate embeddings for query and templates
        query_embedding = embedder.encode(query, normalize_embeddings=True)
        template_embeddings = embedder.encode(QUERY_TEMPLATES, normalize_embeddings=True)

        # Compute cosine similarities
        similarities = np.dot(template_embeddings, query_embedding)
        best_template_idx = np.argmax(similarities)
        best_template = QUERY_TEMPLATES[best_template_idx]

        # Fill template with keywords
        rewritten = best_template.format(关键词=keyword_str)
        logger.info(f"Rewritten query using bge-large-zh-v1.5: {rewritten}")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed with embedder: {e}, returning original query")
        return query

def detect_outliers_iqr(data: List[float]) -> List[float]:
    if not data:
        return []
    arr = np.array(data)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = arr[(arr < lower_bound) | (arr > upper_bound)].tolist()
    return outliers

async def get_recent_files(max_files: int = 5) -> List[str]:
    try:
        if not vector_store_actors:
            logger.warning("No vector store actors available for recent files")
            return []
        indexed_files_all = set()
        for actor_idx, actor in enumerate(vector_store_actors):
            try:
                start_time = time.time()
                indexed_files = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: ray.get(actor.list_indexed_files.remote(), timeout=60)
                )
                duration = time.time() - start_time
                indexed_files_all.update(indexed_files)
                logger.info(f"Retrieved {len(indexed_files)} indexed files from actor {actor_idx} in {duration:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to fetch indexed files from actor {actor_idx} after {duration:.2f}s: {e}")
        recent_files = sorted(indexed_files_all)[:max_files]
        logger.info(f"Retrieved {len(recent_files)} recent files: {recent_files}")
        return recent_files
    except Exception as e:
        logger.error(f"Failed to get recent files: {e}")
        return []

async def stream_error_response(response_id: str, model_name: str, error_message: str):
    error_chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(datetime.now().timestamp()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "delta": {"content": f"Error: {error_message}"},
                "finish_reason": "error"
            }
        ]
    }
    yield f"data: {json.dumps(error_chunk)}\n\n"
    yield "data: [DONE]\n\n"

async def preprocess_csv(file_path: str) -> List[Dict]:
    try:
        logger.info(f"Preprocessing CSV: {file_path}")
        df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')
        ft_model = fasttext.load_model("/models/lid.176.bin")
        text_chunks = []
        for start_idx in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start_idx:start_idx + BATCH_SIZE]
            for _, row in batch.iterrows():
                row_values = []
                for field in df.columns:
                    if pd.notna(row[field]):
                        field_value = str(row[field]).strip()
                        if field_value:
                            prediction = ft_model.predict(field_value.replace("\n", " "), k=1)
                            lang = prediction[0][0].replace("__label__", "")
                            logger.debug(f"Field '{field}' language: {lang}, value: {field_value[:50]}")
                            if lang == "zh":
                                words = jieba.cut(field_value)
                                processed_value = " ".join(words)
                            else:
                                processed_value = field_value
                            row_values.append(f"{field}: {processed_value}")
                if not row_values:
                    logger.debug(f"Skipping empty row at index {row.name}")
                    continue
                text = " ".join(row_values)
                row_metadata = {
                    'row_data': row.to_dict(),
                    'header': df.columns.tolist(),
                    'language': lang
                }
                text_chunks.append({
                    "content": text,
                    "file_name": os.path.basename(file_path),
                    "upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "metadata": row_metadata
                })
        logger.info(f"Preprocessed CSV {file_path}: {len(text_chunks)} text chunks")
        return text_chunks
    except Exception as e:
        logger.error(f"Failed to preprocess CSV {file_path}: {e}")
        return []

async def stream_model_response(url: str, payload: Dict, response_id: str, model_name: str, timeout: int):
    payload["stream"] = True
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.post(url, json=payload, timeout=timeout) as response:
                if response.status != 200:
                    logger.error(f"Streaming request to {url} failed with status {response.status}")
                    async for chunk in stream_error_response(response_id, model_name, f"Request failed with status {response.status}"):
                        yield chunk
                    return
                buffer = ""
                async for data in response.content.iter_chunked(STREAMING_BUFFER_SIZE):
                    buffer += data.decode('utf-8', errors='ignore')
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        if line.startswith("data: "):
                            content = line[6:].strip()
                            if content == "[DONE]":
                                yield f"data: [DONE]\n\n"
                                return
                            try:
                                chunk_data = json.loads(content)
                                chunk = {
                                    "id": response_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(datetime.now().timestamp()),
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": chunk_data.get("response", "")},
                                            "finish_reason": None
                                        }
                                    ]
                                }
                                yield f"data: {json.dumps(chunk)}\n\n"
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse streaming chunk: {content}")
                if buffer:
                    logger.warning(f"Residual buffer after streaming: {buffer[:200]}")
    except Exception as e:
        logger.error(f"Streaming error for {url}: {e}")
        async for chunk in stream_error_response(response_id, model_name, f"Streaming failed: {str(e)}"):
            yield chunk

def validate_metadata(metadata: Dict) -> Dict:
    validated = {}
    for key, value in metadata.items():
        if key == "header" and (value is None or not isinstance(value, (str, list))):
            logger.warning(f"Invalid header in metadata: {value}, setting to empty string")
            validated[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            validated[key] = value
        elif isinstance(value, (list, dict)):
            validated[key] = json.dumps(value, ensure_ascii=False)
        else:
            logger.warning(f"Invalid metadata type for key {key}: {type(value)}")
            validated[key] = str(value)
    return validated

async def generate_response_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"

async def log_request_metrics(endpoint: str, duration: float, status: str, model: str):
    logger.info({
        "endpoint": endpoint,
        "duration": duration,
        "status": status,
        "model": model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

async def multi_model_query(request: QueryRequest):
    query = request.query
    task_type = request.task_type.lower()
    file_names = request.file_names or []
    max_results = min(request.max_results, 10)
    output_format = request.output_format.lower()
    responses = {}
    all_samples = {}
    start_time = time.time()

    try:
        context = ""
        search_results = []

        timeout = extract_timeout(query)
        custom_prompt = extract_custom_prompt(query)
        cleaned_query, extracted_prompt = clean_query(query, custom_prompt)
        final_prompt = extracted_prompt if extracted_prompt else custom_prompt
        logger.info(f"Processed query: cleaned_query='{cleaned_query}', finalApple Music: final_prompt='{final_prompt}', timeout={timeout}s")

        async with await get_aiohttp_session() as session:
            rewritten_query = await rewrite_query(cleaned_query, session)
            keywords = extract_keywords(rewritten_query)
            logger.info(f"Keywords for context adherence: {keywords}")

            extracted_file = extract_file_name(rewritten_query)
            if extracted_file and not file_names:
                file_names = [extracted_file]
                logger.info(f"Extracted file name from query: {file_names}")

            if vector_store_actors:
                logger.info(f"Querying {len(vector_store_actors)} VectorStore actors")
                try:
                    search_results = await hybrid_search(rewritten_query, file_names, max_results)
                    context_parts = []
                    current_length = 0
                    for result in search_results:
                        content = result.get("content", "")
                        file_name = result.get("file_name", "Unknown")
                        upload_time = result.get("upload_timestamp", "Unknown")
                        chunk = f"[{content}] (File: {file_name}, Uploaded: {upload_time})"
                        chunk_length = len(chunk)
                        if current_length + chunk_length > MAX_CONTEXT_CHARS:
                            break
                        context_parts.append(chunk)
                        current_length += chunk_length
                    context = "\n".join(context_parts)
                    logger.info(f"Constructed context (length: {len(context)} chars): {context[:200]}...")
                except Exception as e:
                    logger.error(f"Vector store search failed: {e}")
                    context = "No relevant context found due to search error."
            else:
                logger.warning("No VectorStore actors available, skipping search")
                context = "No vector store actors available."

            if task_type in ["summarize", "compare", "analyze", "anomaly"] and file_names:
                context_lines = []
                if vector_store_actors:
                    for file_name in file_names:
                        file_results = await hybrid_search(rewritten_query, [file_name], 5)
                        current_length = 0
                        file_context = []
                        for result in file_results:
                            upload_time = result.get("upload_timestamp", "Unknown")
                            chunk = f"[{result['content']}] (File: {file_name}, Uploaded: {upload_time})"
                            chunk_length = len(chunk)
                            if current_length + chunk_length > MAX_CONTEXT_CHARS:
                                break
                            file_context.append(chunk)
                            current_length += chunk_length
                        context_lines.extend(file_context)
                    context = "\n".join(context_lines)
                    logger.info(f"Retrieved context for {task_type} (length: {len(context)} chars): {context[:200]}...")
                else:
                    context = "No vector store actors available."

            output_instruction = ""
            if output_format == "json_table":
                output_instruction = 'Return the result in JSON table format: { "data": [{"column1": "value1", "column2": "value2"}, ...] }'
            elif output_format == "json":
                output_instruction = "Return the result in JSON format."
            else:
                output_instruction = "Return the result in plain text."

            default_summarize_prompt = (
                f"Summarize the content of {', '.join(file_names or ['all files'])} in approximately 100 words, "
                f"focusing on the specific data requested in the query: {rewritten_query}. "
                f"Include quantities, dates, or key metrics if available."
            )
            default_compare_prompt = (
                f"Compare the content of {', '.join(file_names)} with respect to: {rewritten_query}. "
                f"Highlight similarities, differences, and key insights in a concise paragraph."
            )
            default_analyze_prompt = (
                f"Perform a statistical or thematic analysis based on: {rewritten_query}. "
                f"Provide counts, trends, or patterns across files, using specific examples."
            )
            default_polish_prompt = (
                f"Polish the text extracted from the context related to: {rewritten_query}. "
                f"Improve its style, grammar, and clarity."
            )
            default_expand_prompt = (
                f"Expand on the content related to: {rewritten_query}. "
                f"Add more details, context, or examples while maintaining the original meaning."
            )
            default_generate_prompt = (
                f"Generate creative text based on the context related to: {rewritten_query}. "
                f"For example, continue a story, write a poem, or create a dialogue."
            )
            default_translate_prompt = (
                f"Translate the text related to: {rewritten_query} into English "
                f"(or specify the target language in the query). Ensure accuracy and natural phrasing."
            )
            default_sentiment_prompt = (
                f"Analyze the sentiment of the text associated with: {rewritten_query}. "
                f"Determine if it is positive, negative, or neutral, and provide a brief explanation."
            )
            default_anomaly_prompt = (
                f"Identify and explain any anomalies in the numerical data related to: {rewritten_query}. "
                f"Use the precomputed outliers and provide insights based on the context."
            )
            default_search_prompt = (
                f"Answer the following query strictly based on the provided context. "
                f"If the context is insufficient to answer any part of the query, "
                f"state \"I lack sufficient information to answer [specific part]\" instead of guessing."
            )

            if query.startswith("@#%") or task_type in ["summarize", "compare", "analyze", "anomaly"]:
                if task_type == "generate_title":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or 'Generate a concise, 3-5 word title with an emoji summarizing the chat history.'}\n\n"
                        f"### Output:\nJSON format: {{ \"title\": \"your concise title here\" }}"
                    )
                elif task_type == "summarize":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_summarize_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "compare":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_compare_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "analyze":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_analyze_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "polish":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_polish_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "expand":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_expand_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "generate":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_generate_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "translate":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_translate_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "sentiment":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_sentiment_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "anomaly":
                    numerical_data = []
                    for result in search_results:
                        content = result.get('content', '')
                        numbers = re.findall(r'\d+\.?\d*', content)
                        numerical_data.extend([float(num) for num in numbers])
                    outliers = detect_outliers_iqr(numerical_data)
                    anomaly_context = f"Extracted numerical data: {numerical_data}\nDetected outliers (using IQR method): {outliers}\nOriginal Context:\n{context}"
                    prompt_template = (
                        f"Context:\n{anomaly_context}\n\n"
                        f"### Task:\n{final_prompt or default_anomaly_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "list_files":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\nList all file names mentioned in the context.\n\n"
                        f"{output_instruction}"
                    )
                else:
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_search_prompt}\n\n"
                        f"{output_instruction}\n\n"
                        f"### Query:\n{rewritten_query}\n"
                    )
            else:
                if task_type == "generate_title":
                    prompt_template = (
                        f"### Task:\n{final_prompt or 'Generate a concise, 3-5 word title with an emoji summarizing the chat history.'}\n\n"
                        f"### Output:\nJSON format: {{ \"title\": \"your concise title here\" }}"
                    )
                elif task_type == "summarize":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_summarize_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "compare":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_compare_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "analyze":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_analyze_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "polish":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_polish_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "expand":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_expand_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "generate":
                    prompt_template = (
                        f"Context:\n{context}\n\n"
                        f"### Task:\n{final_prompt or default_generate_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "translate":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_translate_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "sentiment":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_sentiment_prompt}\n\n"
                        f"{output_instruction}"
                    )
                elif task_type == "anomaly":
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_anomaly_prompt}\n\n"
                        f"{output_instruction}"
                    )
                else:
                    prompt_template = (
                        f"### Task:\n{final_prompt or default_search_prompt}\n\n"
                        f"{output_instruction}\n\n"
                        f"### Query:\n{rewritten_query}\n"
                    )

            payload_gemma = {
                "model": "gemma3:27b-it-q4_K_M",
                "prompt": prompt_template,
                "stream": False,
                "options": {"num_threads": 16, "unload_after_inference": True}
            }
            payload_mixtral = {
                "model": "mixtral:8x7b",
                "prompt": prompt_template,
                "stream": False,
                "max_tokens": 512,
                "options": {"num_threads": 16, "unload_after_inference": True}
            }
            payload_vllm = {
                "model": VLLM_MODEL,
                "prompt": prompt_template,
                "max_tokens": 64 if task_type == "generate_title" else 512,
                "temperature": 0.2,
                "top_p": 0.6
            }

            try:
                tasks = [
                    query_model(session, OLLAMA_URL, payload_gemma, num_samples=NUM_SAMPLES, timeout=max(timeout, GEMMA_TIMEOUT_SECONDS)),
                    query_model(session, OLLAMA_URL, payload_mixtral, num_samples=NUM_SAMPLES, timeout=timeout),
                    query_model(session, VLLM_URL, payload_vllm, num_samples=NUM_SAMPLES, timeout=timeout)
                ]
                gemma_samples, mixtral_samples, vllm_samples = await asyncio.gather(*tasks, return_exceptions=True)

                all_samples["ollama_gemma3:27b-it-q4_K_M"] = gemma_samples if isinstance(gemma_samples, list) else [None]
                all_samples["ollama_mixtral:8x7b"] = mixtral_samples if isinstance(mixtral_samples, list) else [None]
                all_samples["vllm"] = vllm_samples if isinstance(vllm_samples, list) else [None]

            except Exception as e:
                logger.error(f"Error querying models: {e}")
                all_samples = {
                    "ollama_gemma3:27b-it-q4_K_M": [None],
                    "ollama_mixtral:8x7b": [None],
                    "vllm": [None]
                }

            context_adherence_scores = {
                model_name: compute_context_adherence(samples[0], context, keywords) if samples and samples[0] else 0.0
                for model_name, samples in all_samples.items()
            }
            logger.info(f"Context adherence scores: {context_adherence_scores}")

            best_responses = {}
            for model_name, samples in all_samples.items():
                best_response, context_score = select_best_response(samples, context_adherence_scores[model_name], context, keywords)
                best_responses[model_name] = best_response
                if best_response:
                    logger.info(f"Best response for {model_name}: {best_response[:200]}... (context-score: {context_score:.2f})")

            cross_model_scores = compute_cross_model_scores(best_responses)
            logger.info(f"Cross-model scores: {cross_model_scores}")

            final_scores = {}
            for model_name in best_responses:
                context_score = context_adherence_scores.get(model_name, 0.0)
                cross_score = cross_model_scores.get(model_name, 0.0)
                weight_context = 0.6 if context_score < 0.7 else 0.5
                weight_cross = 1.0 - weight_context
                final_scores[model_name] = weight_context * context_score + weight_cross * cross_score
            logger.info(f"Final scores: {final_scores}")

            total_duration = time.time() - start_time
            response_data = {
                "query": cleaned_query,
                "task_type": task_type,
                "context": context[:2000] if context else "",
                "search_results": [
                    {
                        "file_name": r.get("file_name", "Unknown"),
                        "content": r.get("content", ""),
                        "distance": r.get("distance", float("inf")),
                        "relevance_score": r.get("relevance_score", 0.0),
                        "metadata": r.get("metadata", {}),
                        "upload_timestamp": r.get("upload_timestamp", "Unknown")
                    }
                    for r in search_results
                ],
                "best_responses": best_responses,
                "scores": final_scores,
                "context_adherence_scores": context_adherence_scores,
                "cross_model_scores": cross_model_scores,
                "duration": total_duration
            }

            if output_format in ["json", "json_table"]:
                return response_data
            else:
                formatted_response = best_responses.get("vllm", "No valid response generated.")
                return formatted_response

    except Exception as e:
        logger.error(f"Multi-model query failed: {e}", exc_info=True)
        response_id = await generate_response_id()
        await log_request_metrics("/multi_model_query", time.time() - start_time, "error", "fastapi-multi-model")
        return StreamingResponse(
            stream_error_response(response_id, "fastapi-multi-model", f"Internal server error: {str(e)}"),
            media_type="text/event-stream"
        )

@app.post("/multi_model_query", dependencies=[Depends(verify_api_key)])
async def multi_model_query_endpoint(request: QueryRequest):
    response = await multi_model_query(request)
    return response

@app.post("/upload_file", dependencies=[Depends(verify_api_key)])
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = os.path.basename(file.filename)
        if not filename:
            logger.error("No filename provided in upload")
            raise HTTPException(status_code=400, detail="No filename provided")

        supported_extensions = {".txt", ".csv", ".xlsx", ".xls", ".json", ".pdf", ".docx", ".jpg", ".jpeg", ".png", ".ppt", ".pptx"}
        if not any(filename.lower().endswith(ext) for ext in supported_extensions):
            logger.error(f"Unsupported file type: {filename}")
            raise HTTPException(status_code=400, detail="Unsupported file type")

        file_path = os.path.join(container_data_dir, filename)
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await file.read(1024 * 1024):
                await out_file.write(content)

        if not vector_store_actors:
            logger.warning("No VectorStore actors available, file uploaded but not processed")
            return {"message": f"File {filename} uploaded but not processed due to no available actors"}

        actor_idx = assign_actor_for_file(filename, vector_store_actors)
        actor = vector_store_actors[actor_idx]
        logger.info(f"Processing file {filename} with actor vector_store_{actor_idx}")

        vectors_added = 0
        if filename.lower().endswith('.csv'):
            text_chunks = await preprocess_csv(file_path)
            if text_chunks:
                start_time = time.time()
                vectors_added = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ray.get(actor.add_text_chunks.remote(text_chunks), timeout=7200)
                )
                duration = time.time() - start_time
                logger.info(f"Vectorized CSV {filename} with {vectors_added} vectors in {duration:.2f}s")
        else:
            start_time = time.time()
            vectors_added = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: ray.get(
                    actor.add_file.remote(
                        file_path=file_path,
                        metadata={"upload_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                    ),
                    timeout=7200
                )
            )
            duration = time.time() - start_time
            logger.info(f"Vectorized file {filename} with {vectors_added} vectors in {duration:.2f}s")

        logger.info(f"Uploaded and processed {filename}, added {vectors_added} vectors")
        return {"filename": filename, "vectors_added": vectors_added}

    except Exception as e:
        logger.error(f"File upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/upload_chunk", dependencies=[Depends(verify_api_key)])
async def upload_chunk(
    file: UploadFile = File(...),
    chunk_index: int = Form(...),
    total_chunks: int = Form(...),
    x_api_key: str = Header(default=None)
):
    try:
        await verify_api_key(x_api_key)
        file_id = os.path.basename(file.filename)
        if not file_id:
            logger.error("File upload missing filename")
            raise HTTPException(status_code=422, detail="File missing filename")

        temp_path = os.path.join(cache_dir, f"temp_{file_id}")
        logger.info(f"Appending chunk {chunk_index}/{total_chunks} for {file_id} to {temp_path}")
        async with aiofiles.open(temp_path, "ab") as f:
            content = await file.read()
            await f.write(content)
        logger.info(f"Successfully appended chunk {chunk_index}/{total_chunks} for {file_id}")

        return {
            "status": "chunk_received",
            "file_id": file_id,
            "chunk_index": chunk_index
        }
    except HTTPException as e:
        logger.error(f"Upload chunk error: {e}")
        raise
    except Exception as e:
        logger.error(f"Upload chunk error: {e}")
        raise HTTPException(status_code=500, detail=f"Upload chunk failed: {e}")

@app.post("/complete", dependencies=[Depends(verify_api_key)])
async def complete_upload(
    filename: str = Form(...),
    sha256: str = Form(...),
    chunk_hashes: str = Form(...),
    total_chunks: int = Form(...),
    x_api_key: str = Header(default=None)
):
    start_time = time.time()
    duration = 0
    try:
        await verify_api_key(x_api_key)
        filename = os.path.basename(filename)
        temp_path = os.path.join(cache_dir, f"temp_{filename}")
        final_path = os.path.join(cache_dir, filename)
        container_file_path = os.path.join(container_data_dir, filename)

        if not os.path.exists(temp_path):
            logger.error(f"Temporary file {temp_path} not found")
            raise HTTPException(status_code=400, detail="Temporary file not found")

        with open(temp_path, "rb") as f:
            computed_sha256 = hashlib.sha256(f.read()).hexdigest()
        if computed_sha256 != sha256:
            logger.error(f"SHA256 mismatch for {filename}: expected {sha256}, got {computed_sha256}")
            os.remove(temp_path)
            raise HTTPException(status_code=400, detail="SHA256 mismatch")

        shutil.move(temp_path, final_path)
        logger.info(f"Finalized file {filename} at {final_path}")

        vector_count = 0
        if vector_store_actors:
            actor_idx = assign_actor_for_file(filename, vector_store_actors)
            logger.info(f"Processing file {filename} with actor vector_store_{actor_idx}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata = {"upload_timestamp": timestamp, "source": "upload"}
            success = False
            for attempt_actor_idx in [actor_idx] + [(actor_idx + i) % len(vector_store_actors) for i in range(1, len(vector_store_actors))]:
                actor = vector_store_actors[attempt_actor_idx]
                logger.info(f"Attempting vectorization with actor vector_store_{attempt_actor_idx}")
                try:
                    vector_start_time = time.time()
                    if filename.lower().endswith('.csv'):
                        logger.info(f"Preprocessing CSV {container_file_path} for vectorization")
                        text_chunks = await preprocess_csv(container_file_path)
                        if text_chunks:
                            logger.info(f"Generated {len(text_chunks)} text chunks for CSV {filename}")
                            vector_count = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: ray.get(actor.add_text_chunks.remote(text_chunks), timeout=7200)
                            )
                        else:
                            logger.warning(f"No text chunks generated for CSV {filename}")
                    else:
                        vector_count = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: ray.get(actor.add_file.remote(file_path=container_file_path, metadata=metadata), timeout=7200)
                        )
                    duration = time.time() - vector_start_time
                    logger.info(f"Successfully vectorized {container_file_path} with {vector_count} vectors using actor vector_store_{attempt_actor_idx} in {duration:.2f}s")
                    success = True
                    break
                except Exception as e:
                    duration = time.time() - vector_start_time
                    logger.warning(f"Failed to vectorize {filename} with actor vector_store_{attempt_actor_idx} after {duration:.2f}s: {e}")
                    continue
            if not success:
                logger.error(f"All actors failed to vectorize {filename}")
                raise HTTPException(status_code=500, detail="All vector store actors failed to process file")
        else:
            logger.warning("No VectorStore actors available, skipping vectorization")

        await log_request_metrics("/complete", time.time() - start_time, "success", "none")
        return {
            "status": "success",
            "file_id": filename,
            "message": f"Processed {container_file_path} with {vector_count} vectors"
        }
    except HTTPException as e:
        logger.error(f"Complete upload error: {e}")
        await log_request_metrics("/complete", time.time() - start_time, "error", "none")
        raise
    except Exception as e:
        logger.error(f"Complete upload error: {e}")
        await log_request_metrics("/complete", time.time() - start_time, "error", "none")
        raise HTTPException(status_code=500, detail=f"Complete upload failed: {e}")

@app.post("/delete_file", dependencies=[Depends(verify_api_key)])
async def delete_file(filename: str):
    start_time = time.time()
    try:
        filename = os.path.basename(filename)
        file_path = os.path.join(container_data_dir, filename)
        if vector_store_actors:
            actor_idx = assign_actor_for_file(filename, vector_store_actors)
            actor = vector_store_actors[actor_idx]
            logger.info(f"Deleting file {filename} from actor vector_store_{actor_idx}")
            try:
                start_time = time.time()
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ray.get(actor.delete_file.remote(file_path), timeout=120)
                )
                duration = time.time() - start_time
                logger.info(f"Deleted {filename} from VectorStore via actor {actor_idx} in {duration:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to delete {filename} from actor {actor_idx} after {duration:.2f}s: {e}")
        else:
            logger.warning("No VectorStore actors available, skipping VectorStore deletion")

        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Deleted file: {file_path}")
        else:
            logger.warning(f"File {file_path} does not exist")

        await log_request_metrics("/delete_file", time.time() - start_time, "success", "none")
        return {"status": "success", "message": f"Deleted {filename}"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        await log_request_metrics("/delete_file", time.time() - start_time, "error", "none")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reprocess", dependencies=[Depends(verify_api_key)])
async def reprocess(filename: str):
    start_time = time.time()
    try:
        filename = os.path.basename(filename)
        file_path = os.path.join(container_data_dir, filename)
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")

        vector_count = 0
        if vector_store_actors:
            actor_idx = assign_actor_for_file(filename, vector_store_actors)
            actor = vector_store_actors[actor_idx]
            logger.info(f"Reprocessing file {filename} with actor vector_store_{actor_idx}")
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_time = time.time()
            if filename.lower().endswith('.csv'):
                text_chunks = await preprocess_csv(file_path)
                if text_chunks:
                    vector_count = await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda: ray.get(actor.add_text_chunks.remote(text_chunks), timeout=7200)
                    )
            else:
                vector_count = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ray.get(
                        actor.add_file.remote(file_path=file_path, metadata={"upload_timestamp": timestamp, "source": "reprocess"}),
                        timeout=7200
                    )
                )
            duration = time.time() - start_time
            logger.info(f"Reprocessed {file_path} with {vector_count} vectors in {duration:.2f}s")
        else:
            logger.warning("No VectorStore actors available, skipping vectorization")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        async with aiofiles.open(os.path.join(cache_dir, "text.txt"), "a") as f:
            await f.write(f"Reprocessed file: {filename} at {timestamp}\n")

        await log_request_metrics("/reprocess", time.time() - start_time, "success", "none")
        return {
            "status": "success",
            "message": f"Reprocessed {file_path} with {vector_count} vectors"
        }
    except Exception as e:
        logger.error(f"Reprocess error: {e}")
        await log_request_metrics("/reprocess", time.time() - start_time, "error", "none")
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {e}")

@app.get("/list_indexed_files", dependencies=[Depends(verify_api_key)])
async def list_indexed_files():
    start_time = time.time()
    try:
        if not vector_store_actors:
            logger.warning("No vector store actors available")
            await log_request_metrics("/list_indexed_files", time.time() - start_time, "success", "none")
            return {"files": []}

        indexed_files_all = set()
        for actor_idx, actor in enumerate(vector_store_actors):
            try:
                start_time = time.time()
                indexed_files = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: ray.get(actor.list_indexed_files.remote(), timeout=60)
                )
                duration = time.time() - start_time
                indexed_files_all.update(indexed_files)
                logger.info(f"Retrieved {len(indexed_files)} indexed files from actor {actor_idx} in {duration:.2f}s")
            except Exception as e:
                logger.warning(f"Failed to fetch indexed files from actor {actor_idx} after {duration:.2f}s: {e}")

        logger.info(f"Total unique indexed files: {len(indexed_files_all)}")
        await log_request_metrics("/list_indexed_files", time.time() - start_time, "success", "none")
        return {"files": sorted(list(indexed_files_all))}

    except Exception as e:
        logger.error(f"Failed to list indexed files: {e}", exc_info=True)
        await log_request_metrics("/list_indexed_files", time.time() - start_time, "error", "none")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse, dependencies=[Depends(verify_openai_api_key)])
async def chat_completions(request: ChatCompletionRequest, fastapi_request: Request):
    start_time = time.time()
    headers = dict(fastapi_request.headers)
    body = await fastapi_request.json()
    logger.info(f"Received /v1/chat/completions request: headers={headers}, body={body}")

    response_id = await generate_response_id()
    model_name = request.model or "fastapi-multi-model"
    messages = request.messages
    max_tokens = request.max_tokens or 512
    temperature = request.temperature or 0.7
    top_p = request.top_p or 0.9
    stream = request.stream
    created_time = int(datetime.now().timestamp())

    cache_key = f"chat:{hashlib.sha256(json.dumps(body).encode()).hexdigest()}"
    if cache_key in response_cache and not stream:
        logger.info(f"Cache hit for chat completion: {cache_key}")
        cached_response = response_cache[cache_key]
        await log_request_metrics("/v1/chat/completions", time.time() - start_time, "success", model_name)
        return cached_response

    try:
        latest_user_message = None
        for message in reversed(messages):
            if message.get("role") == "user":
                latest_user_message = message.get("content", "")
                break
        if not latest_user_message:
            logger.error("No user message found in request")
            await log_request_metrics("/v1/chat/completions", time.time() - start_time, "error", model_name)
            raise HTTPException(status_code=400, detail="No user message found")

        logger.info(f"Processing latest user query: {latest_user_message}")

        timeout = extract_timeout(latest_user_message)
        custom_prompt = extract_custom_prompt(latest_user_message)
        cleaned_query, extracted_prompt = clean_query(latest_user_message, custom_prompt)
        final_prompt = extracted_prompt if extracted_prompt else custom_prompt
        logger.info(f"Chat completion query: cleaned_query='{cleaned_query}', final_prompt='{final_prompt}', timeout={timeout}s, stream={stream}")

        async with await get_aiohttp_session() as session:
            rewritten_query = await rewrite_query(cleaned_query, session)
            keywords = extract_keywords(rewritten_query)
            logger.info(f"Keywords for context adherence: {keywords}")

            context = ""
            search_results = []
            if vector_store_actors:
                logger.info(f"Querying {len(vector_store_actors)} VectorStore actors for chat completion")
                try:
                    search_results = await hybrid_search(rewritten_query, None, 5)
                    context_parts = []
                    current_length = 0
                    for result in search_results:
                        content = result.get("content", "")
                        file_name = result.get("file_name", "Unknown")
                        upload_time = result.get("upload_timestamp", "Unknown")
                        metadata = validate_metadata(result.get("metadata", {}))
                        chunk = f"[{content}] (File: {file_name}, Uploaded: {upload_time}, Metadata: {json.dumps(metadata)})"
                        chunk_length = len(chunk)
                        if current_length + chunk_length > MAX_CONTEXT_CHARS:
                            break
                        context_parts.append(chunk)
                        current_length += chunk_length
                    context = "\n".join(context_parts)
                    logger.info(f"Constructed context (length: {len(context)} chars): {context[:200]}...")
                except Exception as e:
                    logger.error(f"Vector store search failed: {e}")
                    context = "No relevant context found due to search error."
            else:
                logger.warning("No VectorStore actors available, skipping search")
                context = "No vector store actors available."

            prompt_template = (
                "Context:\n" + context + "\n\n"
                "### Task:\n" + (final_prompt or "Answer the query based on the provided context. If insufficient, state \"I lack sufficient information.\"") + "\n\n"
                "### Query:\n" + rewritten_query + "\n"
            )
            payload_gemma = {
                "model": "gemma3:27b-it-q4_K_M",
                "prompt": prompt_template,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "options": {"num_threads": 16, "unload_after_inference": True}
            }
            payload_mixtral = {
                "model": "mixtral:8x7b",
                "prompt": prompt_template,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "options": {"num_threads": 16, "unload_after_inference": True}
            }
            payload_vllm = {
                "model": VLLM_MODEL,
                "prompt": prompt_template,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }

            if stream:
                async def stream_response():
                    tasks = [
                        stream_model_response(OLLAMA_URL, payload_gemma, response_id, "gemma3:27b-it-q4_K_M", max(timeout, GEMMA_TIMEOUT_SECONDS)),
                        stream_model_response(OLLAMA_URL, payload_mixtral, response_id, "mixtral:8x7b", timeout),
                        stream_model_response(VLLM_URL, payload_vllm, response_id, "vllm", timeout)
                    ]
                    responses = []
                    for task in asyncio.as_completed(tasks):
                        try:
                            async for chunk in await task:
                                responses.append(chunk)
                                yield chunk
                        except Exception as e:
                            logger.error(f"Stream task failed: {e}")
                            async for chunk in stream_error_response(response_id, model_name, f"Stream task failed: {str(e)}"):
                                yield chunk
                    await log_request_metrics("/v1/chat/completions", time.time() - start_time, "success", model_name)

                return StreamingResponse(
                    stream_response(),
                    media_type="text/event-stream",
                    headers={"Cache-Control": "no-cache"}
                )

            # Non-streaming response
            tasks = [
                query_model(session, OLLAMA_URL, payload_gemma, num_samples=1, timeout=max(timeout, GEMMA_TIMEOUT_SECONDS)),
                query_model(session, OLLAMA_URL, payload_mixtral, num_samples=1, timeout=timeout),
                query_model(session, VLLM_URL, payload_vllm, num_samples=1, timeout=timeout)
            ]
            gemma_samples, mixtral_samples, vllm_samples = await asyncio.gather(*tasks, return_exceptions=True)

            all_samples = {
                "ollama_gemma3:27b-it-q4_K_M": gemma_samples if isinstance(gemma_samples, list) else [None],
                "ollama_mixtral:8x7b": mixtral_samples if isinstance(mixtral_samples, list) else [None],
                "vllm": vllm_samples if isinstance(vllm_samples, list) else [None]
            }

            context_adherence_scores = {
                model_name: compute_context_adherence(samples[0], context, keywords) if samples and samples[0] else 0.0
                for model_name, samples in all_samples.items()
            }
            logger.info(f"Context adherence scores: {context_adherence_scores}")

            best_responses = {}
            for model_name, samples in all_samples.items():
                best_response, context_score = select_best_response(samples, context_adherence_scores[model_name], context, keywords)
                best_responses[model_name] = best_response
                if best_response:
                    logger.info(f"Best response for {model_name}: {best_response[:200]}... (context-score: {context_score:.2f})")

            cross_model_scores = compute_cross_model_scores(best_responses)
            logger.info(f"Cross-model scores: {cross_model_scores}")

            final_scores = {}
            for model_name in best_responses:
                context_score = context_adherence_scores.get(model_name, 0.0)
                cross_score = cross_model_scores.get(model_name, 0.0)
                weight_context = 0.6 if context_score < 0.7 else 0.5
                weight_cross = 1.0 - weight_context
                final_scores[model_name] = weight_context * context_score + weight_cross * cross_score
            logger.info(f"Final scores: {final_scores}")

            best_model = max(final_scores, key=lambda x: final_scores[x], default="vllm")
            final_response = best_responses.get(best_model, "No valid response generated.")
            if not final_response:
                logger.error("No valid responses from any model")
                await log_request_metrics("/v1/chat/completions", time.time() - start_time, "error", model_name)
                raise HTTPException(status_code=500, detail="No valid responses generated")

            choices = [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": final_response
                },
                "finish_reason": "stop"
            }]

            response = ChatCompletionResponse(
                id=response_id,
                created=created_time,
                model=model_name,
                choices=choices,
                usage={"prompt_tokens": len(prompt_template.split()), "completion_tokens": len(final_response.split()), "total_tokens": 0}
            )

            response_cache[cache_key] = response
            logger.info(f"Cached response for {cache_key}")

            await log_request_metrics("/v1/chat/completions", time.time() - start_time, "success", model_name)
            return response

    except HTTPException as e:
        logger.error(f"Chat completion error: {e}")
        await log_request_metrics("/v1/chat/completions", time.time() - start_time, "error", model_name)
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        await log_request_metrics("/v1/chat/completions", time.time() - start_time, "error", model_name)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
