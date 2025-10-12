import torch
import torch.nn as nn
import timm

class PetNet(nn.Module):
    """
    Custom model combining a vision backbone (like ViT) 
    with a dense feature head.
    """
    def __init__(self, model_name, out_features, inp_channels, pretrained, num_dense, dropout):
        # NOTE: 'dropout' is received as a keyword argument here
        super().__init__()
        
        self.model = timm.create_model(model_name, pretrained=pretrained, in_chans=inp_channels)
        
        # Determine the input feature size for the head
        try:
            n_features = self.model.num_features
        except AttributeError:
            try:
                n_features = self.model.head.in_features
            except AttributeError:
                raise ValueError(f"Could not determine feature size for model: {model_name}")

        # Replace the final classification layer
        self.model.head = nn.Linear(n_features, 128)
        
        # Define the fully connected head
        self.fc = nn.Sequential(
            nn.Linear(128 + num_dense, 64),
            nn.ReLU(),
            nn.Linear(64, out_features)
        )
        
        # The received 'dropout' argument is used here for the nn.Dropout layer
        self.dropout = nn.Dropout(dropout) 
    
    def forward(self, image, dense):
        embeddings = self.model(image)
        x = self.dropout(embeddings) 
        x = torch.cat([x, dense], dim=1) 
        output = self.fc(x)
        return output