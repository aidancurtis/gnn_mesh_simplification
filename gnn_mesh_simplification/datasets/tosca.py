import os
import glob

import torch
import torch_geometric.transforms
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, Data

import trimesh

torch.serialization.add_safe_globals([getattr])


class TOSCA(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        log=True,
        force_reload=False,
    ):
        self.categories = [
            "cat",
            "centaur",
            "david",
            "dog",
            "gorilla",
            "horse",
            "michael",
            "victoria",
            "wolf",
        ]

        super().__init__(root, transform, pre_transform, pre_filter, log, force_reload)
        self.load(
            self.processed_paths[0],
        )

    @property
    def raw_file_names(self):
        return ["cat0.off"]

    @property
    def processed_file_names(self):
        return f"data.pt"

    def process(self):
        data_list = []
        f2e = torch_geometric.transforms.FaceToEdge(remove_faces=False)

        for i, cat in enumerate(self.categories):
            paths = glob.glob(os.path.join(self.raw_dir, f"{cat}*.off"))
            paths = sorted(paths, key=lambda e: (len(e), e))

            for path in paths:
                mesh = trimesh.load_mesh(path)
                data = f2e(torch_geometric.utils.from_trimesh(mesh))
                data.y = torch.tensor([i], dtype=torch.long)

                if self.pre_filter is not None:
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

        self.save(data_list, self.processed_paths[0])

    def __getitem__(self, idx: int) -> Data:
        data = super().get(idx)
        assert isinstance(data, Data), f"Expected Data object, got {type(data)}"
        return data
