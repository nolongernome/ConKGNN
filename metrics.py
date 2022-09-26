import tensorflow.compat.v1 as tf


def softmax_cross_entropy(preds, labels):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    return tf.reduce_mean(loss)


def accuracy(preds, labels):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    return tf.reduce_mean(accuracy_all)

def contrastive_loss(features, labels, mask,logits_mask,temperature=0.08, base_temperature=0.1):
    """
    Supervised Contrastive Loss
    The distinct batch augumentations must be concatenated in the first axis.
    Currently only supports 2 views per batch.
    Args:
        features: hidden vector of shape [bsz * n_views, ...].
        labels: ground truth of shape [bsz].
    Returns:
        A loss scalar.
    """
    # print(temperature)
    # print(base_temperature)
    features=tf.nn.l2_normalize(features,dim=-1)
    f1,f2=tf.split(features,2,0)
    features = tf.concat([tf.expand_dims(f1,1), tf.expand_dims(f2,1)], 1)
    contrast_count = features.shape[1]
    contrast_feature = tf.concat(tf.unstack(features, axis=1), axis=0)
    anchor_feature = contrast_feature
    # compute logits
    anchor_dot_contrast = tf.math.divide(
        tf.matmul(anchor_feature, tf.transpose(contrast_feature)),
        temperature)

    # for numerical stability
    logits_max = tf.reduce_max(anchor_dot_contrast, axis=1)
    logits = anchor_dot_contrast - logits_max
    logits = tf.cast(logits, dtype=tf.float32)
    mask = mask * logits_mask
    mask = tf.cast(mask, dtype=tf.float32)

    # compute log_prob
    exp_logits = tf.multiply(tf.cast(tf.math.exp(logits), dtype=tf.float32), logits_mask)
    log_prob = logits - tf.cast(tf.math.log(tf.math.reduce_sum(exp_logits, axis=1)), dtype=tf.float32)

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / tf.reduce_sum(mask, axis=1)

    # loss
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    loss = tf.reduce_mean(loss)
    return loss