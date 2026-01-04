from __future__ import annotations

import base64
import io
import time
from typing import Optional

from PIL import Image
import pyspz
import torch
import gc

from config import Settings, settings
from logger_config import logger
from schemas import (
    GenerateRequest,
    GenerateResponse,
    TrellisParams,
    TrellisRequest,
    TrellisResult,
)
from modules.image_edit.qwen_edit_module import QwenEditModule
from modules.background_removal.rmbg_manager import BackgroundRemovalService
from modules.gs_generator.trellis_manager import TrellisService
from modules.utils import (
    secure_randint,
    set_random_seed,
    decode_image,
    to_png_base64,
    save_files,
)

from compare import compare

class GenerationPipeline:
    def __init__(self, settings: Settings = settings):
        self.settings = settings

        # Initialize modules
        self.qwen_edit = QwenEditModule(settings)
        self.rmbg = BackgroundRemovalService(settings)
        self.trellis = TrellisService(settings)

    async def startup(self) -> None:
        """Initialize all pipeline components."""
        logger.info("Starting pipeline")
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)

        await self.qwen_edit.startup()
        self._clean_gpu_memory()
        
        await self.rmbg.startup()
        self._clean_gpu_memory()
        
        await self.trellis.startup()
        self._clean_gpu_memory()

        logger.info("Warming up generator...")
        await self.warmup_generator()
        self._clean_gpu_memory()

        logger.success("Warmup is complete. Pipeline ready to work.")

    async def shutdown(self) -> None:
        """Shutdown all pipeline components."""
        logger.info("Closing pipeline")

        # Shutdown all modules
        await self.qwen_edit.shutdown()
        await self.rmbg.shutdown()
        await self.trellis.shutdown()

        logger.info("Pipeline closed.")

    def _clean_gpu_memory(self) -> None:
        """
        Clean the GPU and CPU memory.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    async def warmup_generator(self) -> None:
        """Warm up the generator with a dummy image to initialize models."""
        temp_image = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buffer = io.BytesIO()
        temp_image.save(buffer, format="PNG")
        temp_image_bytes = buffer.getvalue()
        await self.generate_from_upload(temp_image_bytes, seed=42)

    async def generate_from_upload(self, image_bytes: bytes, seed: int) -> bytes:
        """
        Generate 3D model from uploaded image file and return PLY as bytes.

        Args:
            image_bytes: Raw image bytes from uploaded file

        Returns:
            PLY file as bytes
        """
        # Encode to base64
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # Create request
        request = GenerateRequest(
            prompt_image=image_base64, prompt_type="image", seed=seed
        )

        # Generate
        response_3_views, response_1_views = await self.generate_gs(request)

        # Return binary PLY
        if not response_3_views.ply_file_base64 or not response_1_views.ply_file_base64:
            raise ValueError("PLY generation failed")
        
        # Check time if > 25s then return response_3_views
        GENERATION_TIME_THRESHOLD = 25.0
        logger.info(f"Generation time: {response_3_views.generation_time:.2f}s")
        
        if response_3_views.generation_time > GENERATION_TIME_THRESHOLD:
            logger.info("Generation time exceeds threshold, using 3-view model")
            return response_3_views.ply_file_base64
        
        logger.info("Generation time below threshold, comparing models")
        winner = await compare(image_bytes, response_3_views.ply_file_base64, response_1_views.ply_file_base64)

        if winner == 1:
            logger.info("3-view model selected as winner")
            return response_3_views.ply_file_base64
        else:
            logger.info("1-view model selected as winner")
            return response_1_views.ply_file_base64

    async def generate_gs(self, request: GenerateRequest) -> GenerateResponse:
        """
        Execute full generation pipeline.

        Args:
            request: Generation request with prompt and settings

        Returns:
            GenerateResponse with generated assets
        """
        t1 = time.time()
        logger.info(f"New generation request")

        # Set seed
        if request.seed < 0:
            request.seed = secure_randint(0, 10000)
            set_random_seed(request.seed)
        else:
            set_random_seed(request.seed)

        # Decode input image
        image = decode_image(request.prompt_image)

        # Base prompt template for image editing
        EDIT_PROMPT_TEMPLATE = (
            "Show this object in {view} and make sure it is fully visible. "
            "Turn background neutral solid color contrasting with an object. "
            "Delete background details. Delete watermarks. Keep object colors. "
            "Sharpen image details"
        )

        def edit_and_remove_bg(view_description: str) -> Image.Image:
            """Helper function to edit image and remove background."""
            edited = self.qwen_edit.edit_image(
                prompt_image=image,
                seed=request.seed,
                prompt=EDIT_PROMPT_TEMPLATE.format(view=view_description),
            )
            return self.rmbg.remove_background(edited)

        # Generate multiple views of the image
        image_edited = edit_and_remove_bg("left three-quarters view")
        image_without_background = image_edited

        image_edited_2 = edit_and_remove_bg("right three-quarters view")
        image_without_background_2 = image_edited_2

        image_edited_3 = edit_and_remove_bg("back view")
        image_without_background_3 = image_edited_3

        image_edited_4 = edit_and_remove_bg("side view")
        image_without_background_4 = image_edited_4

        original_image_without_background = self.rmbg.remove_background(image)
        trellis_result_3_views: Optional[TrellisResult] = None
        trellis_result_1_views: Optional[TrellisResult] = None

        # Resolve Trellis parameters from request
        trellis_params: TrellisParams = request.trellis_params

        # 3. Generate the 3D model
        trellis_result_3_views = self.trellis.generate(
            TrellisRequest(
                images=[original_image_without_background, image_without_background_2, image_without_background_3, image_without_background_4],
                seed=request.seed,
                params=trellis_params,
            ),
            threshold=10,
        )

        trellis_result_1_views = self.trellis.generate(
            TrellisRequest(
                images=[original_image_without_background, image_without_background],
                seed=request.seed,
                params=trellis_params,
            ),
            threshold=10,
        )

        # Save generated files
        if self.settings.save_generated_files:
            save_files(
                trellis_result_3_views,
                trellis_result_1_views,
                image,
                image_edited,
                image_without_background,
                image_edited_2,
                image_without_background_2,
                image_edited_3,
                image_without_background_3,
            )

        # Convert to PNG base64 for response (only if needed)
        image_edited_base64 = None
        image_without_background_base64 = None
        if self.settings.send_generated_files:
            image_edited_base64 = to_png_base64(image_edited)
            image_without_background_base64 = to_png_base64(image_without_background)

        t2 = time.time()
        generation_time = t2 - t1

        logger.info(f"Total generation time: {generation_time} seconds")
        # Clean the GPU memory
        self._clean_gpu_memory()

        response_3_views = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result_3_views.ply_file if trellis_result_3_views else None,
            image_edited_file_base64=(
                image_edited_base64 if self.settings.send_generated_files else None
            ),
            image_without_background_file_base64=(
                image_without_background_base64 if self.settings.send_generated_files else None
            ),
        )

        response_1_views = GenerateResponse(
            generation_time=generation_time,
            ply_file_base64=trellis_result_1_views.ply_file if trellis_result_1_views else None,
            image_edited_file_base64=(
                image_edited_base64 if self.settings.send_generated_files else None
            ),
            image_without_background_file_base64=(
                image_without_background_base64 if self.settings.send_generated_files else None
            ),
        )

        logger.success("Generation completed successfully")
        return response_3_views, response_1_views
