class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    self.encoder_layers = [
      nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # 16, 14x14
      nn.ReLU(),
      nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 32, 7x7
      nn.ReLU(),
      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=2, padding=1),  # 64, 4x4
    ]
    self.decoder_layers = [
      nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=7, stride=2, padding=1),  # 32, 7x7
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding = 1),  # 16, 14x14
      nn.ReLU(),
      nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding = 1),  # 1, 28x28
    ]

    self.encoder = nn.Sequential(*self.encoder_layers)
    self.decoder = nn.Sequential(*self.decoder_layers)

  def __repr__(self):
    encoder_str = "\n".join([str(layer) for layer in self.encoder_layers])
    decoder_str = "\n".join([str(layer) for layer in self.decoder_layers])
    return f'\nEncoder Layers:\n{encoder_str}\n' + "-" * 10 + f'\nDecoder Layers: \n{decoder_str}'
  
  def __len__(self):
     return len(self.encoder_layers)+ len(self.decoder_layers)
    
  def __setitem__(self, key, value):
      if key.startswith("encoder"):
        idx = int(key.split("_")[1])
        self.encoder_layers.insert(idx, value)
        print(f"Encoder Modified:\n")
        print("\n".join([str(layer) for layer in self.encoder_layers]))
      
      elif key.startswith("decoder"):
        idx = int(key.split("_")[1])
        self.decoder_layers.insert(idx, value)
        print("\n".join([str(layer) for layer in self.decoder_layers]))

  def forward(self, x):
      return self.decoder(self.encoder(x))


model = Autoencoder()  #initialize model
criterion = nn.MSELoss()  # Loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) # Optimizer
