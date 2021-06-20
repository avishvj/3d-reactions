from torch_geometric.data import Data, Batch

### something to come back to later if necessary
# would have to create CustomDataLoader, CustomBatch, CustomCollater
# then create my own collate() and Batch.from_data_list() funcs
# CustomDataLoader is super simple, the main logic is in CustomCollater which defines the collate() func for the DL
# left this alone because Batch had a fair few funcs and didn't want to cause issues
#   - I think the simple batch each thing works anyway

class CustomBatch(Data):
    def __init__(self, batch = None, ptr):
        pass


def identity_collate(data_list):
    return data_list

class CustomDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size = 1, shuffle = False, collate_fn = identity_collate, **kwargs):
        super(CustomDataLoader, self).__init__(dataset, batch_size, shuffle, collate_fn = identity_collate, **kwargs)
        # change to collate_fn = CustomCollater(follow_batch, exclude_keys)
    

class CustomCollater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys
    
    def __call__(self, batch):
        return self.collate(batch)
    
    def collate(self, batch):
        elem = batch[0]

        # we have Data(4 torch.Tensor and one int)

        if isinstance(elem, Data):
            return Batch.from_data_list(batch, self.follow_batch, self.exclude_keys)


        if isinstance(elem, torch.Tensor):
            pass 