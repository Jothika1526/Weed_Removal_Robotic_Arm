import argparse

import torch
from torch.utils.data import DataLoader

from iknet import IKDataset, IKNet
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kinematics-pose-csv", type=str, default="./dataset/test/kinematics_pose.csv"
    )
    parser.add_argument(
        "--joint-states-csv", type=str, default="./dataset/test/joint_states.csv"
    )
    parser.add_argument("--batch-size", type=int, default=10000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IKNet()
    model.load_state_dict(torch.load("iknet.pth", map_location=torch.device('cpu')))
    model.to(device)
    model.eval()

    dataset = IKDataset(args.kinematics_pose_csv, args.joint_states_csv)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        total_loss += (output - target).norm().item() / args.batch_size
    print(f"Total loss = {total_loss}")

    preds = []
actuals = []

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
    preds.append(output.detach().cpu().numpy())
    actuals.append(target.cpu().numpy())

    preds = np.concatenate(preds)
    actuals = np.concatenate(actuals)

    # Plot first joint angle (you can extend this to more)
    plt.figure(figsize=(6, 6))
    plt.scatter(actuals[:, 0], preds[:, 0], alpha=0.5)
    plt.xlabel("Actual Joint Angle 1")
    plt.ylabel("Predicted Joint Angle 1")
    plt.title("Prediction vs Actual (Joint 1)")
    plt.grid(True)
    plt.plot([actuals[:, 0].min(), actuals[:, 0].max()],
         [actuals[:, 0].min(), actuals[:, 0].max()], 'r--')
    plt.savefig("joint1_pred_vs_actual.png")
    plt.show()
if __name__ == "__main__":
    main()
