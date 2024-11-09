A fork of neosr modified to fit my personal needs and idéas. The code in this fork is **experimental**. If you should decide to use it, you do so at your own risk :grin:

Visit the [neosr repository](https://github.com/neosr-project/neosr/) for the official version. There you'll also find the excellent [Wiki](https://github.com/neosr-project/neosr/wiki) with in-depth information about the configuration options, losses, schedulers and much more.

#### Modifications
- Copying of LQ images to visialization folder disabled per default. It can be re-enabled by adding `copy_lq = true` under `[val]` in your options file.
- Patch size is checked on initialization and, if it can't be safely divided by eight, lowered to a safe value.
- Option to control when validation starts, to enable add `val_start = <iteration-number-to-start>` under `[val]` in your options file. 
- Added padding to the iter count when saving visualization images. So instead of `my-lq-image_5000.png`, `my-lq-image_10000.png` and so on it'll be named `my-lq-image_05000.png`, `my-lq-image_10000.png` and so on.
- Increased metric precision in logs and print out from four to six. Before and after:
    ```
    # dists: 0.7011........ Best: 0.7011 @ 460000 iter
    # dists: 0.7011........ Best: 0.7011 @ 475000 iter
    ```
    ```
    # dists: 0.701114........ Best: 0.701114 @ 460000 iter
    # dists: 0.701159........ Best: 0.701159 @ 475000 iter
    ```

- Added an evaluation option. Works like visualization, but instead of using LQ and GT images it'll run the model currently under training against all images in any folder you specify. Can be useful if you train a model with a specific goal and wish to see if it'll work on images which isn't a part of the dataset. Just like with validation you can set when to start the evaluation and at what frequence to run it. Example:
    ```
    [datasets.eval]
    name = "eval"
    type = "single"
    recursive = false
    dataroot_lq = 'c:/path/to/folder/with/images/to/evaluate'

    [eval]
    eval_start = 10000
    eval_freq = 10000
    ```
