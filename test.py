#%%
import torch 
import numpy as np 
import seaborn as sns


# %%
# create a tensor 
x = torch.tensor(5.5, requires_grad= True)

#%% 
y = x + 10
print(y)
# %%
x = torch.tensor(2.0, requires_grad=True)

# %%
def y_function(val): 
    return(val-3)*(val-6)*(val-4)
# %%
x_range = np.linspace(0, 10, 101)
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y = y_range)
# %%
y = (x-3)*(x-6)*(x-4)
y.backward()
print(x.grad)
# %% second
