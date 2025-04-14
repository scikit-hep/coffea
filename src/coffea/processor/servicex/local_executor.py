# Copyright (c) 2019, IRIS-HEP
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from typing import Callable, Optional

from .executor import Executor, run_coffea_processor


class LocalExecutor(Executor):
    def __init__(self):
        pass

    def get_result_file_stream(self, datasource, title):
        return datasource.stream_result_files(title)

    def run_async_analysis(
        self,
        file_url: str,
        tree_name: Optional[str],
        data_type: str,
        meta_data: dict[str, str],
        process_func: Callable,
        schema,
    ):
        # TODO: Do we need a second routine here? Can we just use this one?
        return self._async_analysis(
            events_url=file_url,
            tree_name=tree_name,
            data_type=data_type,
            meta_data=meta_data,
            process_func=process_func,
            schema=schema,
        )

    async def _async_analysis(
        self, events_url, tree_name, data_type, meta_data, process_func, schema
    ):
        return run_coffea_processor(
            events_url=events_url,
            tree_name=tree_name,
            data_type=data_type,
            meta_data=meta_data,
            proc=process_func,
            schema=schema,
        )
