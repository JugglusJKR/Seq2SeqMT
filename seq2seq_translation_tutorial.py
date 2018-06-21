

######################################################################
# Exercises
# =========
# TODO
# -  Try with a different dataset
#
#    -  Another language pair
#    -  Human → Machine (e.g. IOT commands)
#    -  Chat → Response
#    -  Question → Answer
#
# TODO
# -  Replace the embeddings with pre-trained word embeddings such as word2vec or
#    GloVe
# -  Try with more layers, more hidden units, and more sentences. Compare
#    the training time and results.
# -  If you use a translation file where pairs have two of the same phrase
#    (``I am test \t I am test``), you can use this as an autoencoder. Try
#    this:
#
#    -  Train as an autoencoder
#    -  Save only the Encoder network
#    -  Train a new Decoder for translation from there
#
