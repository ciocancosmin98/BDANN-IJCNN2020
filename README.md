## Environment

You can use podman (or docker if you modify the Makefile) to create a container with the proper environment:

```bash
make pimage
make prun
```

And now you can connect with any terminal to the container or with VSCode's extension for remote containers.

## Dataset

The romanian satire detection dataset can be downloaded from https://drive.google.com/drive/folders/1lw8RCMsFQZ02occFR8OutYySKXvaLugM


## Training

An example of how to run a training session can be seen in "startup.sh" although you first need to
run the "convert.py" script to preprocess the dataset.