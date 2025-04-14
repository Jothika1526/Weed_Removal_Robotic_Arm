import numpy as np
import pandas as pd

class IKDataset:
    def __init__(self, pose_csv, joint_csv, mean=None, std=None, fit_stats=False):
        pose = pd.read_csv(pose_csv)
        joints = pd.read_csv(joint_csv)
        self.x = pose.iloc[:, 3:10].values.astype(np.float32)
        self.y = joints.iloc[:, 8:12].values.astype(np.float32)

        if fit_stats:
            self.mean = self.x.mean(axis=0)
            self.std = self.x.std(axis=0) + 1e-8
        else:
            self.mean = mean
            self.std = std

        self.x = (self.x - self.mean) / self.std

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x[idx], self.y[idx]

class IKNet:
    def __init__(self, input_size=7, output_size=4, hidden_layers=[256, 128, 64, 32], dropout=0.2):
        self.dropout = dropout
        self.layer_sizes = [input_size] + hidden_layers + [output_size]
        self.num_layers = len(self.layer_sizes) - 1

        # Xavier Initialization
        self.weights = [np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * 
                        np.sqrt(2 / (self.layer_sizes[i] + self.layer_sizes[i+1]))
                        for i in range(self.num_layers)]
        self.biases = [np.zeros((1, self.layer_sizes[i+1])) for i in range(self.num_layers)]

        self.m_w, self.v_w = [np.zeros_like(w) for w in self.weights], [np.zeros_like(w) for w in self.weights]
        self.m_b, self.v_b = [np.zeros_like(b) for b in self.biases], [np.zeros_like(b) for b in self.biases]

        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.iteration = 1

    def leaky_relu(self, x): return np.where(x > 0, x, 0.01 * x)
    def leaky_relu_deriv(self, x): return np.where(x > 0, 1.0, 0.01)

    def forward(self, x, training=True):
        self.a = [x]
        self.z = []
        self.masks = []

        for i in range(self.num_layers - 1):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            a = self.leaky_relu(z)

            if training:
                mask = (np.random.rand(*a.shape) > self.dropout).astype(np.float32)
                a *= mask / (1 - self.dropout)
            else:
                mask = np.ones_like(a)

            self.z.append(z)
            self.a.append(a)
            self.masks.append(mask)

        out = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(out)
        return out

    def loss(self, pred, target, delta=1.0):
        diff = pred - target
        abs_diff = np.abs(diff)
        mask = abs_diff < delta
        loss = np.where(mask, 0.5 * diff**2, delta * (abs_diff - 0.5 * delta))
        return np.mean(loss)

    def backward(self, y_pred, y_true, lr):
        grad = 2 * (y_pred - y_true) / y_true.shape[0]
        grads_w, grads_b = [], []

        for i in reversed(range(self.num_layers)):
            dw = np.dot(self.a[i].T, grad)
            db = np.sum(grad, axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                grad = np.dot(grad, self.weights[i].T)
                grad *= self.leaky_relu_deriv(self.z[i - 1])
                grad *= self.masks[i - 1]

        # Gradient Clipping
        max_grad_norm = 1.0
        grads_w = [np.clip(g, -max_grad_norm, max_grad_norm) for g in grads_w]
        grads_b = [np.clip(g, -max_grad_norm, max_grad_norm) for g in grads_b]

        for i in range(self.num_layers):
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * grads_w[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (grads_w[i] ** 2)
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

            m_w_hat = self.m_w[i] / (1 - self.beta1 ** self.iteration)
            v_w_hat = self.v_w[i] / (1 - self.beta2 ** self.iteration)
            m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.iteration)
            v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.iteration)

            self.weights[i] -= lr * m_w_hat / (np.sqrt(v_w_hat) + self.eps)
            self.biases[i] -= lr * m_b_hat / (np.sqrt(v_b_hat) + self.eps)

        self.iteration += 1

    def predict(self, x): return self.forward(x, training=False)

def train(model, train_data, val_data, epochs=150, batch_size=128, lr=0.0005):
    best_val_loss = float('inf')
    patience, wait = 15, 0
    warmup_epochs = 5

    for epoch in range(1, epochs + 1):
        np.random.shuffle(train_data)
        train_loss = []

        adjusted_lr = lr * min(1.0, epoch / warmup_epochs)

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            x = np.stack([item[0] for item in batch])
            y = np.stack([item[1] for item in batch])
            pred = model.forward(x, training=True)
            loss = model.loss(pred, y)
            model.backward(pred, y, adjusted_lr)
            train_loss.append(loss)

        val_loss = evaluate(model, val_data, batch_size)
        print(f"Epoch {epoch:03}: Train Loss = {np.mean(train_loss):.6f}, Val Loss = {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                lr *= 0.5
                wait = 0
                print(f"🔻 Learning rate reduced to {lr:.6f}")

        if best_val_loss < 0.009:
            print(f"\n🎯 Target loss reached: {best_val_loss:.6f}. Early stopping!")
            break

def evaluate(model, data, batch_size=128):
    losses = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        x = np.stack([item[0] for item in batch])
        y = np.stack([item[1] for item in batch])
        pred = model.predict(x)
        losses.append(model.loss(pred, y))
    return np.mean(losses)

if __name__ == "__main__":
    train_ds = IKDataset("./dataset/train/kinematics_pose.csv", "./dataset/train/joint_states.csv", fit_stats=True)
    test_ds = IKDataset("./dataset/test/kinematics_pose.csv", "./dataset/test/joint_states.csv", mean=train_ds.mean, std=train_ds.std)

    split = int(0.8 * len(train_ds))
    train_data = [train_ds[i] for i in range(split)]
    val_data = [train_ds[i] for i in range(split, len(train_ds))]
    test_data = [test_ds[i] for i in range(len(test_ds))]

    model = IKNet()
    train(model, train_data, val_data, epochs=150, batch_size=128, lr=0.0005)

    test_loss = evaluate(model, test_data)
    print(f"\n✅ Final Test Loss: {test_loss:.6f}")
