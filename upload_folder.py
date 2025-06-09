#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import requests
import time
import shutil
import logging
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
from fastapi import FastAPI, HTTPException, Form
from queue import Queue
import threading
import sys
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to INFO for production, DEBUG for detailed troubleshooting
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/opt/ai-platform/upload_folder.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Log startup
logger.info("Starting upload_folder.py")

# Check for required dependencies
try:
    import uvicorn
except ImportError as e:
    logger.error("Missing uvicorn dependency: %s", str(e))
    sys.exit(1)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError as e:
    logger.error("Missing watchdog dependency: %s", str(e))
    sys.exit(1)

try:
    import requests
except ImportError as e:
    logger.error("Missing requests dependency: %s", str(e))
    sys.exit(1)

UPLOAD_DIR = "/home/j/upload"
PROCESSING_DIR = "/tmp/processing"
FAILED_DIR = "/tmp/failed"
API_URL = "http://192.168.0.125:8081/upload_chunk"
DELETE_API_URL = "http://192.168.0.125:8081/delete_file"
COMPLETE_API_URL = "http://192.168.0.125:8081/complete"
HEADERS = {"X-API-Key": "your-secret-key"}
UPLOAD_TIMEOUT = int(os.getenv("UPLOAD_TIMEOUT", 600))  # Increased to 600 seconds
DELETE_TIMEOUT = int(os.getenv("DELETE_TIMEOUT", 30))
CHUNK_SIZE = 1024 * 1024  # 1MB chunks
supported_extensions = (".pdf", ".docx", ".jpg", ".png", ".xlsx", ".xls", ".csv", ".txt", ".ppt", ".pptx", ".md", ".json")
FILE_STABLE_TIMEOUT = 30  # Seconds to wait for file size to stabilize
FILE_STABLE_INTERVAL = 1  # Seconds between size checks

# Ensure directories exist
for directory in [PROCESSING_DIR, FAILED_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# FastAPI app for health check
app = FastAPI()
logger.info("FastAPI app initialized")

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}

def check_service():
    try:
        response = requests.get("http://192.168.0.125:8081/health", headers=HEADERS, timeout=5)
        logger.debug("Health check: %s", response.status_code)
        return response.status_code == 200
    except requests.RequestException as e:
        logger.error("Health check failed: %s", str(e))
        return False

def wait_for_file_stable(file_path):
    """Wait until the file size stabilizes to ensure it's fully written."""
    start_time = time.time()
    last_size = -1
    while time.time() - start_time < FILE_STABLE_TIMEOUT:
        try:
            current_size = os.path.getsize(file_path)
            logger.debug(f"Checking {file_path} size: {current_size} bytes")
            if current_size == last_size and current_size > 0:
                logger.info(f"File {file_path} size stabilized at {current_size} bytes")
                return current_size
            last_size = current_size
            time.sleep(FILE_STABLE_INTERVAL)
        except OSError as e:
            logger.warning(f"Error checking {file_path} size: {str(e)}")
            time.sleep(FILE_STABLE_INTERVAL)
    logger.error(f"File {file_path} did not stabilize within {FILE_STABLE_TIMEOUT} seconds")
    raise Exception(f"File {file_path} size did not stabilize")

def upload_file(file_path, retry_with_new_name=False):
    original_file_id = os.path.basename(file_path)
    file_id = f"{uuid.uuid4().hex}_{original_file_id}" if retry_with_new_name else original_file_id
    processing_path = os.path.join(PROCESSING_DIR, file_id)
    retries = 3
    backoff_factor = 5
    chunk_hashes = []

    if not check_service():
        logger.error("Service unavailable, pausing upload")
        return False

    try:
        # Wait for file to stabilize
        original_size = wait_for_file_stable(file_path)
        logger.info(f"Original file {file_path} size: {original_size} bytes")

        # Compute full file SHA256
        with open(file_path, "rb") as f:
            full_sha256 = hashlib.sha256(f.read()).hexdigest()
        logger.info(f"File {file_path} full SHA256: {full_sha256}")

        # Copy file to processing directory
        logger.info(f"Copying {file_path} to {processing_path} for processing")
        shutil.copy(file_path, processing_path)
        logger.info(f"Removing original file {file_path}")
        os.remove(file_path)

        # Verify copied file size
        copied_size = os.path.getsize(processing_path)
        logger.info(f"Copied file {processing_path} size: {copied_size} bytes")
        if original_size != copied_size:
            logger.error(f"Size mismatch for {file_id}: original {original_size}, copied {copied_size}")
            raise Exception(f"File size mismatch after copy: {copied_size} vs {original_size}")

        # Read file and send chunks
        file_size = copied_size
        total_chunks = (file_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        logger.info(f"Preparing to upload {file_id}, size: {file_size} bytes, total chunks: {total_chunks}")

        with open(processing_path, "rb") as f:
            for chunk_index in range(total_chunks):
                chunk_data = f.read(CHUNK_SIZE)
                chunk_size = len(chunk_data)
                chunk_hash = hashlib.sha256(chunk_data).hexdigest()
                chunk_hashes.append(chunk_hash)
                logger.debug(f"Chunk {chunk_index}: size={chunk_size}, SHA256={chunk_hash}")

                for attempt in range(retries):
                    try:
                        files = {"file": (file_id, chunk_data, "application/octet-stream")}
                        data = {"chunk_index": chunk_index, "total_chunks": total_chunks}
                        logger.debug(f"Sending chunk {chunk_index} for {file_id} with total_chunks: {total_chunks}")
                        response = requests.post(
                            API_URL,
                            files=files,
                            data=data,
                            headers=HEADERS,
                            proxies={"http": None, "https": None},
                            timeout=UPLOAD_TIMEOUT,
                        )
                        if response.status_code == 200:
                            logger.info(f"Successfully uploaded chunk {chunk_index} for {file_id}: {response.json()}")
                            break
                        else:
                            error_message = f"Attempt {attempt + 1}/{retries} - Failed to upload chunk {chunk_index} for {file_id}: {response.status_code} - {response.text}"
                            logger.warning(error_message)
                            if attempt == retries - 1:
                                logger.error(f"All retries failed for chunk {chunk_index} of {file_id}")
                                if "Total chunks mismatch" in response.text and not retry_with_new_name:
                                    logger.info(f"Retrying upload for {original_file_id} with new filename: {file_id}")
                                    raise Exception(f"Total chunks mismatch, triggering retry: {response.text}")
                                raise Exception(f"Failed to upload chunk {chunk_index}: {response.text}")
                            time.sleep(backoff_factor * (2 ** attempt))
                    except Exception as e:
                        logger.error(f"Attempt {attempt + 1}/{retries} - Error uploading chunk {chunk_index} for {file_id}: {str(e)}")
                        if attempt == retries - 1:
                            logger.error(f"All retries failed for chunk {chunk_index} of {file_id}")
                            raise
                        time.sleep(backoff_factor * (2 ** attempt))

        # Send completion request
        try:
            chunk_hashes_str = ",".join(chunk_hashes)  # Convert to comma-separated string
            complete_data = {
                "filename": file_id,
                "sha256": full_sha256,
                "chunk_hashes": chunk_hashes_str,
                "total_chunks": total_chunks
            }
            logger.debug(f"Sending complete request for {file_id} with data: {complete_data}")
            logger.debug(f"Chunk hashes for {file_id}: {chunk_hashes_str}")
            response = requests.post(
                COMPLETE_API_URL,
                data=complete_data,
                headers=HEADERS,
                proxies={"http": None, "https": None},
                timeout=UPLOAD_TIMEOUT
            )
            if response.status_code == 200:
                logger.info(f"Completed upload for {file_id}: {response.json()}")
                try:
                    os.remove(processing_path)
                    logger.info(f"Deleted {processing_path} after successful upload")
                except Exception as e:
                    logger.warning(f"Failed to delete {processing_path}: {str(e)}")
                return True
            else:
                logger.error(f"Failed to complete upload for {file_id}: {response.status_code} - {response.text}")
                raise Exception(f"Complete upload failed: {response.text}")
        except Exception as e:
            logger.error(f"Error completing upload for {file_id}: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Failed to process {file_id}: {str(e)}")
        # Move to failed directory
        failed_path = os.path.join(FAILED_DIR, file_id)
        try:
            if os.path.exists(processing_path):
                shutil.move(processing_path, failed_path)
                logger.info(f"Moved {processing_path} to {failed_path}")
        except Exception as e2:
            logger.error(f"Failed to move {processing_path} to {failed_path}: {str(e2)}")
        # Retry with a new filename if total chunks mismatch
        if "Total chunks mismatch" in str(e) and not retry_with_new_name:
            new_file_id = f"{uuid.uuid4().hex}_{original_file_id}"
            logger.info(f"Retrying upload for {original_file_id} with new filename: {new_file_id}")
            # Create a temporary copy for retry
            temp_path = os.path.join(PROCESSING_DIR, f"temp_{original_file_id}")
            shutil.copy(file_path, temp_path)
            result = upload_file(temp_path, retry_with_new_name=True)
            try:
                os.remove(temp_path)
            except Exception as e2:
                logger.warning(f"Failed to delete temp file {temp_path}: {str(e2)}")
            return result
        return False

class UploadHandler(FileSystemEventHandler):
    def __init__(self, upload_queue):
        self.upload_queue = upload_queue
        self.recently_moved = {}
        self.active_uploads = set()  # Track files being uploaded

    def on_created(self, event):
        if event.is_directory:
            logger.debug("Ignoring directory creation: %s", event.src_path)
            return
        file_path = event.src_path
        file_name = os.path.basename(file_path)
        if file_name.lower().endswith(supported_extensions):
            if file_name in self.active_uploads:
                logger.info(f"File {file_name} is already being uploaded, skipping")
                return
            logger.info(f"Detected new file: {file_path}")
            try:
                wait_for_file_stable(file_path)
                self.recently_moved[file_name] = time.time()
                self.active_uploads.add(file_name)
                self.upload_queue.put((file_path, file_name))
            except Exception as e:
                logger.error(f"Failed to queue {file_name}: {str(e)}")
        else:
            logger.debug("Ignoring unsupported file: %s", file_name)

    def on_deleted(self, event):
        if event.is_directory:
            logger.debug("Ignoring directory deletion: %s", event.src_path)
            return
        file_name = os.path.basename(event.src_path)
        if file_name in self.recently_moved:
            moved_time = self.recently_moved[file_name]
            if time.time() - moved_time < 15:
                logger.info("Ignoring recent moved file deletion: %s", file_name)
                return
        logger.info(f"Detected file deletion: %s", file_name)
        processing_path = os.path.join(PROCESSING_DIR, file_name)
        if not os.path.exists(processing_path):
            try:
                response = requests.post(
                    DELETE_API_URL,
                    json={"filename": file_name},
                    headers=HEADERS,
                    proxies={"http": None, "https": None},
                    timeout=DELETE_TIMEOUT
                )
                logger.info(f"Notified deletion of {file_name}: {response.status_code}")
            except Exception as e:
                logger.error(f"Failed to notify deletion of {file_name}: {str(e)}")

def process_upload_queue(upload_queue, upload_handler):
    while True:
        try:
            file_path, file_name = upload_queue.get()
            logger.info(f"Processing upload for {file_name}")
            if upload_file(file_path):
                logger.info(f"File {file_name} uploaded successfully")
            else:
                logger.error(f"Failed to upload {file_name}")
            upload_queue.task_done()
            # Remove from active uploads
            upload_handler.active_uploads.discard(file_name)
        except Exception as e:
            logger.error(f"Error processing upload queue: {str(e)}")
            upload_handler.active_uploads.discard(file_name)
            upload_queue.task_done()

def monitor_folder():
    upload_queue = Queue()
    event_handler = UploadHandler(upload_queue)
    observer = Observer()
    observer.schedule(event_handler, UPLOAD_DIR, recursive=False)
    observer.start()
    logger.info("Started monitoring directory: %s", UPLOAD_DIR)

    threading.Thread(target=process_upload_queue, args=(upload_queue, event_handler), daemon=True).start()

    try:
        while True:
            time.sleep(10)
    except Exception as e:
        logger.error("Error monitoring %s: %s", UPLOAD_DIR, str(e))
    finally:
        observer.stop()
        observer.join()

if __name__ == "__main__":
    logger.info("Starting upload_folder service")
    threading.Thread(target=monitor_folder, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8082)
