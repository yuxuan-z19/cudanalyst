import asyncio
import os
from typing import override

import dacite

from ..helper.ckpt import iter_program_json
from .intervene import IntervenePipe, IntervenePipeAsync, PromptCtx, Record


class ReplayPipe(IntervenePipe):
    @staticmethod
    def _record_to_ctx(data: dict):
        record = dacite.from_dict(Record, data)
        return PromptCtx(record.id, record.prompt, record.old_profile, record.reports)

    @override
    async def build_all_contexts(self, input_ckpt_dir, concurrency=8):
        contexts = []
        for _, data in iter_program_json(input_ckpt_dir):
            contexts.append(self._record_to_ctx(data))
        return contexts


class ReplayPipeAsync(ReplayPipe, IntervenePipeAsync):
    @override
    async def produce_contexts(
        self,
        input_ckpt_dir: os.PathLike,
        ctx_queue: asyncio.Queue,
        ctx_map: dict[str, PromptCtx],
        llm_concurrency: int = 8,
    ):
        for _, data in iter_program_json(input_ckpt_dir):
            ctx = self._record_to_ctx(data)
            ctx_map[ctx.id.pid] = ctx
            await ctx_queue.put(ctx)

        for _ in range(llm_concurrency):
            await ctx_queue.put(None)
