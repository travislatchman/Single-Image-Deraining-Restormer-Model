import numpy as np
import torch
import matplotlib.pyplot as plt

class DerainModel(torch.nn.Module):
    def __init__(self):
        super(DerainModel, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, 7, padding=(3, 3)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(8, 8, 7, padding=(3, 3)),
            torch.nn.PReLU(),
            torch.nn.Conv2d(8, 1, 7, padding=(3, 3)),
            torch.nn.PReLU()
        )
 
    def forward(self, x):
        decoded = self.conv(x)
        return decoded

path = ["rain100H", "rain100L"]
for rain in path:
    noisy = plt.imread("data/" + rain + "/test/rain/norain-1.png")
    clean = plt.imread("data/" + rain + "/test/norain/norain-1.png")
    denoised_im = np.zeros(noisy.shape)
    for i in range(3):
        noisy_tensor = torch.from_numpy(noisy[:, :, i]).unsqueeze(0).unsqueeze(0)
        clean_tensor = torch.from_numpy(clean[:, :, i]).unsqueeze(0).unsqueeze(0)

        cnn = DerainModel()
        loss = torch.nn.MSELoss()
        opt = torch.optim.Adam(cnn.parameters(), lr=0.005)
        max_epoch = 1000
        
        for epoch in range(max_epoch):
            output = cnn(noisy_tensor)
            err = loss(output, clean_tensor)
            err.backward()
            opt.step()
            opt.zero_grad()
            
            denoised = np.squeeze(output.detach().numpy())
        denoised_im[:, :, i] = denoised

plt.subplot(1, 3, 1)
plt.imshow(clean)
plt.subplot(1, 3, 2)
plt.imshow(noisy)
plt.subplot(1, 3, 3)
plt.imshow(denoised_im)