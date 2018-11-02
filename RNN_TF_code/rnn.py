import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from utils import Vocab


RESET_AFTER = 50
class Config(object):
    """This section is hyper parameter definition
       The important hyper parameters are size of the word embedding used , no of training epochs and early stopping criteria defined
    """
    embed_size = 35
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 20
    lr = 0.01
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)


class RNN_Model():

    def load_data(self):
        """Data loading section"""
        self.train_data, self.dev_data, self.test_data = tr.simplified_data(700, 100, 200)

        # build vocab from training data
        self.vocab = Vocab()
        train_sents = [t.get_words() for t in self.train_data]
        self.vocab.construct(list(itertools.chain.from_iterable(train_sents)))

    def inference(self, tree, predict_only_root=False):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        if predict_only_root:
            node_tensors = node_tensors[tree.root]
        else:
            node_tensors = [tensor for node, tensor in node_tensors.items() if node.label!=2]
            node_tensors = tf.concat(node_tensors, 0)
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        '''
        Model parameters: 
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        
        '''
        with tf.variable_scope('Composition'):
            embedding = tf.get_variable("embedding", [len(self.vocab), self.config.embed_size])
            W1 = tf.get_variable("W1", [2* self.config.embed_size, self.config.embed_size])
            b1 = tf.get_variable("b1", [1, self.config.embed_size])

           
        with tf.variable_scope('Projection'):
        
            U = tf.get_variable("U", [self.config.embed_size, self.config.label_size])
            bs = tf.get_variable("bs", [1, self.config.label_size])

           

    def add_model(self, node):
        """This is the actual part where a recursive model is being built in a bottom-up manner where a parent node is constructed by using a composition function on the children nodes
        """
        with tf.variable_scope('Composition', reuse=True):
            
            embedding = tf.get_variable("embedding")
            W1 = tf.get_variable("W1")
            b1 = tf.get_variable("b1")
            

        node_tensors = dict()
        curr_node_tensor = None
        if node.isLeaf:
            
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, self.vocab.encode(node.word)), axis= 0)
           
        else:
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))
            node_input = tf.concat([node_tensors[node.left], node_tensors[node.right]], 1)
            
            curr_node_tensor = tf.nn.relu(tf.matmul(node_input, W1) + b1)

            
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
        """
        """
        logits = None
        
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable("U")
            bs = tf.get_variable("bs")
        logits = tf.matmul(node_tensors, U) + bs
        
        return logits

    def loss(self, logits, labels):
        """The loss function used is a softmax cross entropy function with logits
        """
        loss = None
       
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable("W1")
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable("U")
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)) + self.config.l2 * (tf.nn.l2_loss(W1)+tf.nn.l2_loss(U))
        
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.
         The original optimizer used in the github code is GradientDescentOptimizer and I tried some other optimizers but that did
		 not significantly improve performance
    
        """
        train_op = None
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.config.lr)
        train_op = optimizer.minimize(loss)

        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        
        predictions = tf.argmax(y, axis=1)

        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        losses = []
        for i in range(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree, True)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    if get_loss:
                        root_label = tree.root.label
                        loss = sess.run(self.loss(logits, [root_label]))
                        losses.append(loss)
                    results.append(root_prediction)
        return results, losses

    def run_epoch(self, new_model = False, verbose=True):
        step = 0
        loss_history = []
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.global_variables_initializer()
                    sess.run(init)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s.temp'%self.config.model_name)
                for _ in range(RESET_AFTER):
                    if step>=len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree)
                    labels = [l for l in tree.labels if l!=2]
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    loss_history.append(loss)
                    if verbose:
                        sys.stdout.write('\r{} / {} :    loss = {}'.format(
                            step, len(self.train_data), np.mean(loss_history)))
                        sys.stdout.flush()
                    step+=1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s.temp'%self.config.model_name)
        train_preds, _ = self.predict(self.train_data, './weights/%s.temp'%self.config.model_name)
        val_preds, val_losses = self.predict(self.dev_data, './weights/%s.temp'%self.config.model_name, get_loss=True)
        train_labels = [t.root.label for t in self.train_data]
        val_labels = [t.root.label for t in self.dev_data]
        train_acc = np.equal(train_preds, train_labels).mean()
        val_acc = np.equal(val_preds, val_labels).mean()

        print()
        print('Training acc (only root node): {}'.format(train_acc))
        print('Validation acc (only root node): {}'.format(val_acc))
        print('Confusion matrix:')
        print(self.make_conf(train_labels, train_preds))
        print(self.make_conf(val_labels, val_preds))
        return train_acc, val_acc, loss_history, np.mean(val_losses)

    def train(self, verbose=True):
        complete_loss_history = []
        train_acc_history = []
        val_acc_history = []
        prev_epoch_loss = float('inf')
        best_val_loss = float('inf')
        best_val_epoch = 0
        stopped = -1
        for epoch in range(self.config.max_epochs):
            print('epoch %d'%epoch)
            if epoch==0:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch(new_model=True)
            else:
                train_acc, val_acc, loss_history, val_loss = self.run_epoch()
            complete_loss_history.extend(loss_history)
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)

            #lr annealing
            epoch_loss = np.mean(loss_history)
            if epoch_loss>prev_epoch_loss*self.config.anneal_threshold:
                self.config.lr/=self.config.anneal_by
                print('annealed lr to %f'%self.config.lr)
            prev_epoch_loss = epoch_loss

            # save if model has improved on val
            if val_loss < best_val_loss:
                 best_val_loss = val_loss
                 best_val_epoch = epoch

            # if model has not improved for a while stop
            if epoch - best_val_epoch > self.config.early_stopping:
                stopped = epoch
                #break
        if verbose:
                sys.stdout.write('\r')
                sys.stdout.flush()

        print('\n\nstopped at %d\n'%stopped)
        return {
            'loss_history': complete_loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
            }

    def make_conf(self, labels, predictions):
        confmat = np.zeros([2, 2])
        for l,p in zip(labels, predictions):
            confmat[l, p] += 1
        return confmat

def plot_figure(data, title, savepath, labels=[]):
    """I have retained the plotting functions from the original code
    """
    
    plt.figure()
    plt.plot(data)
    plt.title(title)
    if len(labels) == 2:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.savefig(savepath)
    plt.show()
    

def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Sentiment
    Analysis network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    stats = model.train(verbose=True)
    print('Training time: {}'.format(time.time() - start_time))

    plot_figure(stats['loss_history'], 'Loss history', "loss_history.png",
        ['Iteration', 'Loss'])
    plot_figure(stats['train_acc_history'], 'Training accuracy history',
        "train_history.png", ['Iteration', 'Accuracy'])
    plot_figure(stats['val_acc_history'], 'Validation accuracy history',
        "validation_history.png", ['Iteration', 'Accuracy'])

    print('Test')
    print('=-=-=')
    predictions, _ = model.predict(model.test_data, './weights/%s.temp'%model.config.model_name)
    labels = [t.root.label for t in model.test_data]
    test_acc = np.equal(predictions, labels).mean()
    print('Test acc: {}'.format(test_acc))

if __name__ == "__main__":
        test_RNN()
