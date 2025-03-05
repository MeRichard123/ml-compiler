# Emulating Lua Execution through different Recurrent Neural Network Architectures: A Comparative Study in Example-Based Learning.

Project Proposal: [here](./Proposal.pdf)

## Model Performances
### Simple models, trained on only alice in the wonderland text
| model | inference |
| ---- | ----- |
| ![Standard RNN](./images/RNN_lossPlot.png) | _*(20/01/2025)*_ Very noisy, but eventually learns. Could benefit from stopping sooner. |
| ![GRU](./images/GRU_lossPlot.png) | _*(20/01/2025)*_ Does well can stop it at 2000 iterations. |
| ![LSTM](./images/LSTM_lossPlot.png) |  _*(20/01/2025)*_ Less noisy, but not tuned well. |

### Models Trained on Code pre AST
| model | inference |
| ---- | ----- |
| ![RNN](./images/RNN_lossPlot_code.png)| *(17/02/25)* |
| ![GRU](./images/gru-code.png) | *(16/02/25)* GRU trained with code data with batch processing enabled. Even more <br /> trained over 15000 iterations|
| ![LSTM](./images/LSTM_lossPlot_code.png)| *(17/02/25)* |
| ![GRU Min](./images/gru-code-min.png) | *(16/02/25)* MiniGRU trained with code data a lot smoother and less noisy. Even more <br /> trained over 15000 iterations|
| ![LSTM Min](./images/LSTM_lossPlot_code_min.png) |*(16/02/25)* mini LSTM trained with code data with batch processing enabled. Even more <br /> trained over 15000 iterations |