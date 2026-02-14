
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import math

EPS = 1e-7
PI = math.pi

def cxcywh_to_xyxy(box):
    cx, cy, w, h = tf.split(box, 4, axis=-1)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = tf.concat([x1, y1, cx + w / 2, cy + h / 2], axis=-1) # correction: split returns tensors, w/2 is tensor. 
    # Wait, the original notebook code was:
    # x2 = cx + w / 2
    # y2 = cy + h / 2
    # return tf.concat([x1, y1, x2, y2], axis=-1)
    # let's stick to original exact code to avoid breaking weights
    
    x2 = cx + w / 2
    y2 = cy + h / 2
    return tf.concat([x1, y1, x2, y2], axis=-1)

def build_full_backbone(input_shape):
    base = applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    # We assume weights are loaded later, so trainable status here matters less for inference, 
    # but good to keep consistent.
    for l in base.layers:
        l.trainable = False

    c3 = base.get_layer("conv3_block4_out").output
    c4 = base.get_layer("conv4_block6_out").output
    c5 = base.get_layer("conv5_block3_out").output

    p5 = layers.Conv2D(256, 1, name="p5")(c5)
    p4 = layers.Add()([
        layers.UpSampling2D()(p5),
        layers.Conv2D(256, 1)(c4)
    ])
    p3 = layers.Add()([
        layers.UpSampling2D()(p4),
        layers.Conv2D(256, 1)(c3)
    ])

    return models.Model(base.input, [p3, p4, p5], name="Full_Backbone")

def build_roi_backbone(input_shape):
    base = applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )
    for l in base.layers:
        l.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)

    return models.Model(base.input, x, name="ROI_Backbone")

def multiscale_head(p3, p4, p5, name):
    def block(x):
        x = layers.Conv2D(256, 1, activation="relu")(x)
        return layers.GlobalAveragePooling2D()(x)

    f3 = block(p3)
    f4 = block(p4)
    f5 = block(p5)

    x = layers.Concatenate(name=name)([f3, f4, f5])
    x = layers.Dense(256, activation="relu")(x)
    return x

def cross_view_gate(cc, mlo):
    channels = cc.shape[-1]
    if channels is None:
         # In eager execution or some versions this might be needed, 
         # but usually for Functional API graph construction it's fine.
         # The notebook code:
         # if channels is None: raise ValueError(...)
         pass

    gate = layers.Dense(int(channels), activation="sigmoid")(cc + mlo)
    cc  = layers.Multiply()([cc, gate])
    mlo = layers.Multiply()([mlo, gate])
    return cc, mlo

class ViewConsistencyLayer(layers.Layer):
    def __init__(self, weight=0.2, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def call(self, cc, mlo):
        loss = tf.reduce_mean(tf.square(cc - mlo))
        self.add_loss(self.weight * loss)
        return cc, mlo
    
    def get_config(self):
        config = super().get_config()
        config.update({"weight": self.weight})
        return config

def build_final_model():
    # ===== Inputs =====
    cc_full  = layers.Input((512,512,3), name="cc_full")
    cc_roi   = layers.Input((224,224,3), name="cc_roi")
    mlo_full = layers.Input((512,512,3), name="mlo_full")
    mlo_roi  = layers.Input((224,224,3), name="mlo_roi")

    # ===== Backbones =====
    full_net = build_full_backbone((512,512,3))
    roi_net  = build_roi_backbone((224,224,3))

    # ===== FULL features =====
    cc_p3, cc_p4, cc_p5   = full_net(cc_full)
    mlo_p3, mlo_p4, mlo_p5 = full_net(mlo_full)

    # ===== ROI features =====
    cc_roi_f  = roi_net(cc_roi)
    mlo_roi_f = roi_net(mlo_roi)

    # ===== Multi-scale =====
    cc_feat  = multiscale_head(cc_p3, cc_p4, cc_p5, "cc_multiscale")
    mlo_feat = multiscale_head(mlo_p3, mlo_p4, mlo_p5, "mlo_multiscale")

    # ===== Inject ROI =====
    cc_feat  = layers.Concatenate()([cc_feat, cc_roi_f])
    mlo_feat = layers.Concatenate()([mlo_feat, mlo_roi_f])

    cc_feat  = layers.Dense(256, activation="relu")(cc_feat)
    mlo_feat = layers.Dense(256, activation="relu")(mlo_feat)

    # ===== Cross-view gate =====
    cc_feat, mlo_feat = cross_view_gate(cc_feat, mlo_feat)

    # ===== View consistency =====
    cc_feat, mlo_feat = ViewConsistencyLayer(0.2)(cc_feat, mlo_feat)

    # ===== Fusion =====
    fused = layers.Concatenate()([cc_feat, mlo_feat])
    fused = layers.Dense(512, activation="relu")(fused)
    fused = layers.Dropout(0.3)(fused)

    # ===== Heads =====
    cls = layers.Dense(1, activation="sigmoid", name="cancer_cls")(fused)
    
    bbox_raw = layers.Dense(4,activation="sigmoid")(fused)
    bbox_out = layers.Lambda(cxcywh_to_xyxy,name="bbox")(bbox_raw)


    return models.Model(
        inputs={
            "cc_full": cc_full,
            "cc_roi": cc_roi,
            "mlo_full": mlo_full,
            "mlo_roi": mlo_roi
        },
        outputs={
            "cancer_cls": cls,
            "bbox": bbox_out
        },
        name="MultiView_CBIS"
    )

# Loss functions needed for loading the model if we load with compile=True
# However, usually for inference we can load with compile=False. 
# But let's include them just in case or if we need to compile.

def biou(y_true, y_pred):
    b1 = cxcywh_to_xyxy(y_true)
    b2 = cxcywh_to_xyxy(y_pred)

    x1 = tf.maximum(b1[..., 0], b2[..., 0])
    y1 = tf.maximum(b1[..., 1], b2[..., 1])
    x2 = tf.minimum(b1[..., 2], b2[..., 2])
    y2 = tf.minimum(b1[..., 3], b2[..., 3])

    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union = area1 + area2 - inter + EPS
    return tf.reduce_mean(inter / union)

def b_ciou(y_true, y_pred):
    b1 = cxcywh_to_xyxy(y_true)
    b2 = cxcywh_to_xyxy(y_pred)

    x1 = tf.maximum(b1[..., 0], b2[..., 0])
    y1 = tf.maximum(b1[..., 1], b2[..., 1])
    x2 = tf.minimum(b1[..., 2], b2[..., 2])
    y2 = tf.minimum(b1[..., 3], b2[..., 3])

    inter = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])
    union = area1 + area2 - inter + EPS

    iou = inter / union

    # center distance
    c1x = (b1[..., 0] + b1[..., 2]) / 2
    c1y = (b1[..., 1] + b1[..., 3]) / 2
    c2x = (b2[..., 0] + b2[..., 2]) / 2
    c2y = (b2[..., 1] + b2[..., 3]) / 2

    center_dist = tf.square(c1x - c2x) + tf.square(c1y - c2y)

    # enclosing box
    enc_x1 = tf.minimum(b1[..., 0], b2[..., 0])
    enc_y1 = tf.minimum(b1[..., 1], b2[..., 1])
    enc_x2 = tf.maximum(b1[..., 2], b2[..., 2])
    enc_y2 = tf.maximum(b1[..., 3], b2[..., 3])

    enc_diag = tf.square(enc_x2 - enc_x1) + tf.square(enc_y2 - enc_y1) + EPS

    # aspect ratio penalty
    w1 = b1[..., 2] - b1[..., 0]
    h1 = b1[..., 3] - b1[..., 1]
    w2 = b2[..., 2] - b2[..., 0]
    h2 = b2[..., 3] - b2[..., 1]

    v = (4.0 / (PI ** 2)) * tf.square(
        tf.atan(w1 / (h1 + EPS)) - tf.atan(w2 / (h2 + EPS))
    )

    alpha = v / (1 - iou + v + EPS)

    ciou = iou - center_dist / enc_diag - alpha * v
    return ciou

class BBoxCIoULoss(tf.keras.losses.Loss):
    def __init__(self, l1_weight=1.0, ciou_weight=1.0, name="bbox_ciou_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        self.l1_weight = l1_weight
        self.ciou_weight = ciou_weight

    def call(self, y_true, y_pred):
        l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

        ciou_score = b_ciou(y_true, y_pred) 
        ciou_loss = 1.0 - ciou_score

        return self.l1_weight * l1 + self.ciou_weight * ciou_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "l1_weight": self.l1_weight,
            "ciou_weight": self.ciou_weight
        })
        return config
