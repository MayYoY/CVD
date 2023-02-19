from torchvision import transforms


class TrainConfig:
    record_path = "./aug_record.csv"
    folds = [1, 2, 4, 5]  # [2, 3, 4, 5]
    tasks = []
    sources = [1, 2, 3]

    trans = transforms.Compose([transforms.ToTensor(), 
                                transforms.RandomVerticalFlip(0.5),
                                # transforms.RandomHorizontalFlip(0.5),
                                transforms.Resize(size=(320, 320))])
    VerticalFlip = True

    batch_size = 8  # 2
    lr = 0.0005
    num_epochs = 20
    device = ""


class TestConfig:
    record_path = "./record.csv"
    folds = [3]  # [1]
    tasks = []
    sources = [1, 2, 3]

    trans = transforms.Compose([transforms.ToTensor(), 
                                transforms.Resize(size=(320, 320))])
    VerticalFlip = False

    batch_size = 5  # 5
    device = ""  # cpu

    post = "fft"
    diff = False
    detrend = True
