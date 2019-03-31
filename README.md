# ahana-learner
My ML project for 6th Sem :wink: :ok_hand:

## What it's about

This is a project which I took during 6th Semester to make a Recursive Neural Network(RNN) train on a textual dataset,
and then use it to output real words, given some input. It uses a type of RNNs called Long Short Term Memory(LSTM)
Networks to do the job.

Here are some useful pointers:
  * [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
  * [The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

## Run n Load

First of all, install these dependencies: 

```
pip3 install numpy tensorflow
```

Then once loaded, run the bot using Python3

```
$ python3 bot.py
```

The bot connects to a channel on [Freenode](http://freenode.net/) using a inbuilt Bot class.
To send messages to the bot type this.
```
[insert-bot-name] hey
```

Auto generated messages using the bot's trained models are sent. Models are trained on the 
tensorflow Shakespeare text.

To train a new model, try changing this 2 lines:
```
EPOCHS=15

history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])
```
EPOCHS being the number of times you want to train the model.
