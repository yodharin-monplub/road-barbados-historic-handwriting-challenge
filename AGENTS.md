## Task Difficulty Assessment

Before beginning each task:

1. Rate its difficulty relative to the current model:
   - 1 — Trivial
   - 2 — Easy
   - 3 — Moderate
   - 4 — Hard
   - 5 — Near the model's practical limit
2. State the main source of difficulty.
3. Recommend a different model or reasoning effort if appropriate.
4. Reassess the rating after inspecting the repository.

I am competing in "R.O.A.D. Barbados Historic Handwriting Challenge"
link "https://zindi.world/competitions/road-barbados-historic-handwriting-challenge"
can you get me a winning prize, doesn't have to be first place.
Work in this directory. create many models starting with model1 model2 model3...
in each model, create a readme.txt

## Hardware limitation 

Has GPU and CUDA on local pc = TRUE.
renting gpu on vast.ai is possible
openrouter is aviable.

## Cloud spending behaviour
Option A: Prioritize speed, try to rent fast GPU. Train fast and when done shut down the instance.
Option B: Prioritize cost, choose option with lower cost, with speed as a secondary factor.
Current option:A
Maximun number of instances allow:1

## goal 

goal: keep pushing for higher score. Aim for both good CV score and public leaderboard score 

## Behavior 

if the model is running, you can hibernate and wait. monitor every 20 minutes.
Keep running and email me every milesstone or if you need human input, if I disagree I will tell you to stop.
Make the session history link to the folder, not to the pc. 
Treat kaggle Gpu quota as precious because it's shared with other competitions. When submit, Just test on the dummy data.
Permission to rent a instance without asking = True
Permission to submit without asking = True (email me the csv file and I will submit for you.)

## Directory format 

Lay the directory as follow 

"Data" folder with the competition data and all external data splitted into subfolder. 

"Model" folder and in that contain many subfolder call model1, model2 model3 .... for each model 1,2,3 in each model (model 1,2,3), I want the source code (require), readme.txt (require) and score.txt(optional) if the score is available. Add other file as you see fit. Make sure that If the prediction and pipeline is reproducible even if I loss access to the claude code session. 

"Other" folder. Containing .env file python .env folder and all other file that you created. 

In the root directory, not in any folder, let there be only AGENTS.md 

## Communication method 

i have created a email for you to communicate with me. If you need human input, Sent me an email to yodharinmonplub@actuarialtutor.org For Example, if you need me to point to a direction. I will reply the email or type the prompt. email.txt contains the username and password Email me if you make progress. Please check email every 3 minutes. 

## Try the following for better score 

if the competition rule allow,use external data and pretrain model. 

Read the discussion forum in the competition website for help. 

Search the internet for help.

## Backup

everyday, once per day when the cpu load is low, backup the project to github. The fine-grained token is in .env and named "GITHUB_FINEGRAIN_ACCESSTOKEN"

