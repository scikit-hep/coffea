# Copyright (c) 2021, IRIS-HEP
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


from collections.abc import AsyncGenerator
from typing import Optional

from func_adl import ObjectStream, find_EventDataset
from servicex import ServiceXDataset, StreamInfoPath, StreamInfoUrl


class DataSource:
    def __init__(
        self,
        query: ObjectStream,
        metadata: dict[str, str] = {},
        datasets: list[ServiceXDataset] = [],
    ):
        self.query = query
        self.metadata = metadata
        self.schema = None
        self.datasets = datasets

    async def _get_query(self) -> str:
        """Return the qastle query.

        Note: To do this we have to forward-cast the object: by design, not all `func_adl`
        queries are `ServiceX` queries. But this library only works with datasets that are
        based in `ServiceX`. Thus some duck typing occurs in this method.
        """
        event_dataset_ast = find_EventDataset(self.query.query_ast)
        event_dataset = event_dataset_ast._eds_object  # type: ignore
        if not hasattr(event_dataset, "return_qastle"):
            raise Exception(
                f"Base func_adl query {str(event_dataset)} does not have a way to generate qastle!"
            )
        event_dataset.return_qastle = True  # type: ignore
        return await self.query.value_async()

    async def stream_result_file_uris(
        self, title: Optional[str] = None
    ) -> AsyncGenerator[tuple[str, str, StreamInfoUrl], None]:
        """Launch all datasources off to servicex

        Yields:
            Tuple[str, StreamInfoUrl]: List of data types and url's to process
        """
        qastle = await self._get_query()

        # TODO: Make this for loop parallel
        for dataset in self.datasets:
            data_type = dataset.first_supported_datatype(["parquet", "root"])
            if data_type == "root":
                async for file in dataset.get_data_rootfiles_uri_stream(
                    qastle, title=title, as_signed_url=True
                ):
                    yield (data_type, dataset.dataset_as_name, file)
            elif data_type == "parquet":
                async for file in dataset.get_data_parquet_uri_stream(
                    qastle, title=title, as_signed_url=True
                ):
                    yield (data_type, dataset.dataset_as_name, file)
            else:
                raise Exception(
                    f"This dataset ({str(dataset)}) supports unknown datatypes"
                )

    async def stream_result_files(
        self, title: Optional[str] = None
    ) -> AsyncGenerator[tuple[str, str, StreamInfoPath], None]:
        """Launch all datasources at once off to servicex

        Yields:
            Tuple[str, StreamInfoPath]: List of data types and file paths to process
        """
        qastle = await self._get_query()

        # TODO: Make this for loop parallel
        for dataset in self.datasets:
            data_type = dataset.first_supported_datatype(["parquet", "root"])
            if data_type == "root":
                async for file in dataset.get_data_rootfiles_stream(
                    qastle, title=title
                ):
                    yield (data_type, dataset.dataset_as_name, file)
            elif data_type == "parquet":
                async for file in dataset.get_data_parquet_stream(qastle, title=title):
                    yield (data_type, dataset.dataset_as_name, file)
            else:
                raise Exception(
                    f"This dataset ({str(dataset)}) supports unknown datatypes"
                )
