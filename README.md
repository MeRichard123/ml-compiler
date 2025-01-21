# Emulating Lua Execution through different Recurrent Neural Network Architectures: A Comparative Study in Example-Based Learning.

Project Proposal: [here](./Proposal.pdf)

## Model Performances
| model | inference |
| ---- | ----- |
| ![Standard RNN](./images/RNN_lossPlot.png) | Very noisy, but eventually learns. Could benefit from stopping sooner. |
| ![GRU](./images/GRU_lossPlot.png) | Does well can stop it at 2000 iterations. |
| ![LSTM](./images/LSTM_lossPlot.png) | Less noisy, but not tuned well. |