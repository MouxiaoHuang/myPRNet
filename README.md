## This is a personal implementation of PRNet 

*keep updating*

- *Notice:* This project is based on ***tensorflow***. And ***pytorch version*** would be implemented and uploaded if possible.

---

### 1. Environment required

> - tensorflow-gpu == 1.13.1 (or higher version except from tf2)
> - numpy == 1.17.4
> - matplotlib
> - opencv-python
> - scikit-image
> - scipy
>
> 

### 2. Introductions of some files

- ***Data***

> net-data: put trained model into it for testing
>
> trainData: save *trainDataLabel.txt* (you need to generate your own training data), which contains the paths of training data
>
> uv-data: mask

- ***checkpoint***

> where the trained model (obtained by *mytrain.py*) saved

### 3. Training steps

- Generate training data (size: about 180 GB)

1. see [jnulzl/PRNet](https://github.com/jnulzl/PRNet) 
2. run *9_generate_prnet_trainset_300WLP.py*

- Run *mytrain.py*

***NOTICE:*** be careful of **all the PATHS** used in .py files, you MUST modify them by yourself

### 4. Test

- you would obtain your own PRNet-model in the file *checkpoint* 
- test your model to see its performance (see the ***Usage*** part of [YadiraF/PRNet](https://github.com/YadiraF/PRNet))
