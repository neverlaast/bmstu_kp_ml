# run.py
import argparse
import os

from src.train import train_model
from src.eval import evaluate_model


def main():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---------- train ----------
    train_parser = subparsers.add_parser("train", help="Train model")

    train_parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Путь к корню с данными (где лежит FullDataSet_PD-BioStampRC21 и Clinic_*.csv)",
    )
    train_parser.add_argument(
        "--clinic_file",
        type=str,
        default="Clinic_DataPDBioStampRCStudy.csv",
        help="Имя клинического файла относительно data_root",
    )
    train_parser.add_argument("--window_sec", type=float, default=10.0)
    train_parser.add_argument("--stride_sec", type=float, default=5.0)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    train_parser.add_argument("--max_train_patients", type=int, default=None)
    train_parser.add_argument("--max_val_patients", type=int, default=None)
    # FIX: параметры сплита — при 34 пациентах 0.20/0.20 даёт ~20 train / 7 val / 7 test
    train_parser.add_argument("--val_size", type=float, default=0.20,
                              help="Доля пациентов для val (default: 0.20)")
    train_parser.add_argument("--test_size", type=float, default=0.20,
                              help="Доля пациентов для test (default: 0.20)")

    # ---------- eval ----------
    eval_parser = subparsers.add_parser("eval", help="Evaluate model")

    eval_parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Путь к корню с данными (где лежит FullDataSet_PD-BioStampRC21 и Clinic_*.csv)",
    )
    eval_parser.add_argument(
        "--clinic_file",
        type=str,
        default="Clinic_DataPDBioStampRCStudy.csv",
        help="Имя клинического файла относительно data_root",
    )
    eval_parser.add_argument("--window_sec", type=float, default=10.0)
    eval_parser.add_argument("--stride_sec", type=float, default=5.0)
    eval_parser.add_argument("--batch_size", type=int, default=32)
    eval_parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    eval_parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Путь к .pt файлу с моделью; если не задан, берём checkpoints/best_model.pt",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_model(
            data_root=args.data_root,
            clinic_filename=args.clinic_file,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            batch_size=args.batch_size,
            lr=args.lr,
            num_epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
            max_train_patients=args.max_train_patients,
            max_val_patients=args.max_val_patients,
            val_size=args.val_size,
            test_size=args.test_size,
        )

    elif args.command == "eval":
        ckpt = args.checkpoint_path
        if ckpt is None:
            ckpt = os.path.join(args.checkpoint_dir, "best_model.pt")
        evaluate_model(
            data_root=args.data_root,
            clinic_filename=args.clinic_file,
            checkpoint_path=ckpt,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            batch_size=args.batch_size,
        )


if __name__ == "__main__":
    main()
