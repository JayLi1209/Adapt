Before each run, do a command similar to:
```
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader | sort -t, -k2 -rn | head -8; echo "stray:"; pgrep -f sweep_dirichlet_layers.py | wc -l
```
to see the avaliable cores out there. Then, use several cores that are the most avaliable.

If you were asked to run an experiment, no discount and no truncation.
Pretraining the model should always prioritize training the transitions close to the goal more.

Whenever I asked you to explain the experiment setting, explain at least the following:
1. What is the nature of the change (at what timestep changed to what)
2. What is the architecture (number of layers, dim of each layer, etc.) of the problem
3. What are all the hyperparameters used (especially, discount factor, alpha)
4. After how many steps the env truncates, and how many trials does it face such truncation
5. If it's FrozenLake, what is its reward structure, 

The default setting for frozenlake:
no truncation, no discount, [1,0,0] -> [0.7,0.15,0.15] at ts 0, dirichelet head, averaged over 100 trials with error bars. For now, K_FORGET = 1, and report error bars and averaged dbar across trials on the first time the env detected change.

The default setting for pendulum:
mass 1 -> 4 at ts 0, stop at episode 200, discount 0.99, gaussian head, averaged 100 trials with error bars. Use the learned-reward head as default approach.

After each prompt, if there's new file added, tell me the file names; tell me the line numbers, roughly, of the existing modified files. 