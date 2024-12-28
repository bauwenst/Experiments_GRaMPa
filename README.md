# Experiments_GRaMPa
Code for the GRaMPa paper's experiments.

## Dependencies
To be able to run the scripts in this repo, you should have installed the following packages:
```shell
pip install "tktkt[github]  @ git+https://github.com/bauwenst/TkTkT"
pip install "lamoto[github] @ git+https://github.com/bauwenst/LaMoTO"
pip install "fiject         @ git+https://github.com/bauwenst/fiject"
```
Due to the incompetence of some HuggingFace team members, the `transformers` training framework and the DeBERTa
implementation have not been stable for about 2 to 3 months, so you're better off using this older version:
```
pip install transformers==4.45.2
```