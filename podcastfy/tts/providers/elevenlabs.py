"""ElevenLabs TTS provider implementation with chunking."""

from elevenlabs import client as elevenlabs_client
from ..base import TTSProvider
from typing import List
import re
import logging
from io import BytesIO
from pydub import AudioSegment

logger = logging.getLogger(__name__)

class ElevenLabsTTS(TTSProvider):
    def __init__(self, api_key: str, model: str = "eleven_multilingual_v2", timeout: float = 300.0):
        """
        Initialize ElevenLabs TTS provider.
        
        Args:
            api_key (str): ElevenLabs API key
            model (str): Model name to use. Defaults to "eleven_multilingual_v2"
            timeout (float): Request timeout in seconds. Defaults to 300.0
        """
        logger.info(f"Initializing ElevenLabs TTS with model: {model}")
        # Configure the client with custom timeout
        self.client = elevenlabs_client.ElevenLabs(
            api_key=api_key,
            timeout=timeout  # Set a longer timeout (5 minutes)
        )
        self.model = model
        
    def split_text_into_chunks(self, text: str, max_chars: int = 2000) -> List[str]:
        """
        Split text into smaller chunks at sentence boundaries.
        
        Args:
            text (str): Text content to convert to speech
            max_chars (int): Maximum characters per chunk
            
        Returns:
            List[str]: List of text chunks
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        # Split on sentence boundaries (period, exclamation mark, or question mark followed by space or newline)
        sentences = re.split(r'([.!?]+(?:\s+|$))', text)
        sentences = [s for s in sentences if s]
        
        current_chunk = ""
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            # Get the punctuation that follows this sentence if there is one
            separator = sentences[i + 1] if i + 1 < len(sentences) else ""
            complete_sentence = sentence + separator
            
            if len(current_chunk) + len(complete_sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = complete_sentence
                else:
                    # If a single sentence is too long, split at word boundaries
                    words = complete_sentence.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk) + len(word) + 1 > max_chars:
                            chunks.append(temp_chunk.strip())
                            temp_chunk = word
                        else:
                            temp_chunk += " " + word if temp_chunk else word
                    current_chunk = temp_chunk
            else:
                current_chunk += complete_sentence
                
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        logger.info(f"Split text into {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            logger.debug(f"Chunk {i+1} length: {len(chunk)} characters")
            
        return chunks
        
    def merge_audio_bytes(self, audio_chunks: List[bytes]) -> bytes:
        """
        Merge multiple audio chunks into a single audio file.
        
        Args:
            audio_chunks (List[bytes]): List of audio data
            
        Returns:
            bytes: Combined audio data
        """
        if not audio_chunks:
            return b""
        
        if len(audio_chunks) == 1:
            return audio_chunks[0]
        
        try:
            # Initialize for storing valid audio segments
            valid_segments = []
            
            for i, chunk in enumerate(audio_chunks):
                try:
                    # Skip empty chunks
                    if not chunk or len(chunk) == 0:
                        logger.warning(f"Skipping empty chunk {i}")
                        continue
                    
                    # Convert bytes to AudioSegment
                    segment = AudioSegment.from_file(BytesIO(chunk), format="mp3")
                    if len(segment) > 0:
                        valid_segments.append(segment)
                        logger.debug(f"Successfully processed chunk {i}, duration: {len(segment)}ms")
                    else:
                        logger.warning(f"Zero-length segment in chunk {i}")
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
            
            if not valid_segments:
                raise RuntimeError("No valid audio chunks to merge")
            
            # Merge valid chunks
            combined = valid_segments[0]
            for segment in valid_segments[1:]:
                combined = combined + segment
            
            # Export to bytes
            output = BytesIO()
            combined.export(
                output,
                format="mp3",
                codec="libmp3lame",
                bitrate="256k"
            )
            
            result = output.getvalue()
            logger.info(f"Successfully merged {len(valid_segments)} audio chunks into {len(result)} bytes")
            return result
            
        except Exception as e:
            logger.error(f"Audio merge failed: {str(e)}", exc_info=True)
            # Fallback to first chunk if merging fails
            if audio_chunks:
                return audio_chunks[0]
            raise RuntimeError(f"Failed to merge audio chunks and no valid fallback found: {str(e)}")
    
    def generate_audio(self, text: str, voice: str, model: str, voice2: str = None) -> bytes:
        """Generate audio using ElevenLabs API with chunking for long texts."""
        logger.info(f"Generating audio for text of length {len(text)} characters")
        
        try:
            # Split text into chunks to avoid timeouts with large requests
            chunks = self.split_text_into_chunks(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            # Generate audio for each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}, length: {len(chunk)} characters")
                try:
                    audio = self.client.generate(
                        text=chunk,
                        voice=voice,
                        model=model
                    )
                    
                    # Collect all bytes from the stream
                    chunk_bytes = b''.join(data for data in audio if data)
                    
                    if chunk_bytes:
                        audio_chunks.append(chunk_bytes)
                        logger.info(f"Successfully generated audio for chunk {i+1}, size: {len(chunk_bytes)} bytes")
                    else:
                        logger.warning(f"Received empty audio for chunk {i+1}")
                        
                except Exception as e:
                    logger.error(f"Error generating audio for chunk {i+1}: {str(e)}")
                    # Continue with other chunks even if one fails
                    continue
            
            # If we have multiple chunks, merge them
            if len(audio_chunks) > 1:
                logger.info(f"Merging {len(audio_chunks)} audio chunks")
                return self.merge_audio_bytes(audio_chunks)
            elif len(audio_chunks) == 1:
                return audio_chunks[0]
            else:
                raise RuntimeError("No audio chunks were successfully generated")
                
        except Exception as e:
            logger.error(f"Error in generate_audio: {str(e)}", exc_info=True)
            raise
        
    def get_supported_tags(self) -> List[str]:
        """Get supported SSML tags."""
        return ['lang', 'p', 'phoneme', 's', 'sub']