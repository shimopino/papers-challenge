import torch


class GANLoss:
    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samples, fake_samples, height, alpha):
        raise NotImplementedError("dis loss method has not been implemented")

    def gen_loss(self, real_samples, fake_samples, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


class ConditionalGANLoss:
    def __init__(self, dis):
        self.dis = dis

    def dis_loss(self, real_samples, fake_samples, labels, height, alpha):
        raise NotImplementedError("dis loss method has not been implemented")

    def gen_loss(self, real_samples, fake_samples, labels, height, alpha):
        raise NotImplementedError("gen_loss method has not been implemented")


class StandardGAN(GANLoss):
    def __init__(self, dis):
        from torch.nn import BCEWithLogitsLoss

        super().__init__(dis)

        # define the criterion and activation used for object
        self.criterion = BCEWithLogitsLoss()

    def dis_loss(self, real_samps, fake_samps, height, alpha):
        # small assertion:
        assert (
            real_samps.device == fake_samps.device
        ), "Real and Fake samples are not on the same device"

        # device for computations:
        device = fake_samps.device

        # predictions for real images and fake images separately :
        r_preds = self.dis(real_samps, height, alpha)
        f_preds = self.dis(fake_samps, height, alpha)

        # calculate the real loss:
        real_loss = self.criterion(
            torch.squeeze(r_preds), torch.ones(real_samps.shape[0]).to(device)
        )

        # calculate the fake loss:
        fake_loss = self.criterion(
            torch.squeeze(f_preds), torch.zeros(fake_samps.shape[0]).to(device)
        )

        # return final losses
        return (real_loss + fake_loss) / 2

    def gen_loss(self, _, fake_samps, height, alpha):
        preds, _, _ = self.dis(fake_samps, height, alpha)
        return self.criterion(
            torch.squeeze(preds), torch.ones(fake_samps.shape[0]).to(fake_samps.device)
        )
