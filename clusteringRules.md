# Category Clustering

After seperating the scenes, we do the following checks:

- If the frames within a scene are changing a lot,the scene is in Category II
- We apply a mask of one plate like in assignment 1:  
    1. If we have one continuous, large enough output: Category I
    2. If we have 2 continuous, large enough outputs: Category III
    3. If we do not have any large enough, continuous outputs: Category IV