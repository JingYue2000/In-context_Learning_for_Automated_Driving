# In-context_Learning_for_Automated_Driving

The code in this repository is based on the paper Reward Design with Language Models. 

This repository contains the prompts that we used for each domain as well as code to train an RL agent with an LLM in the loop using those prompts. Each domain (Ultimatum Game, Matrix Games, DealOrNoDeal) has a separate directory and will need a separate conda/virtual environment. 

Please check out the READMEs in each directory for more information on how to run things.

# Model Setup

- We use GPT3 for our experiments. You will need to have an API key from them saved in a file named MY_KEY.txt.
- Use requirement.txt for model environment setup：
  ```pip install -r requirements.txt```

# Model Work Flow

![image](https://github.com/JingYue2000/In-context_Learning_for_Automated_Driving/blob/main/Framework_fig.png)

# Conservative Model and Aggressive Model

![image](https://github.com/JingYue2000/In-context_Learning_for_Automated_Driving/blob/main/casestudy430.png)

An example of conservative model：

![Conservative Model](Results/conservative.gif)


An example of aggressive model：

![Aggressive Model](Results/aggressive.gif)
