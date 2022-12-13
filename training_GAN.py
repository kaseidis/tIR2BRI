from numpy import real
import torch

import torch.nn as nn


class Discriminator(nn.Module):
    """Model check if the grayscale image is generated by model
    """

    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()

        model1 = [nn.Conv2d(1, 32, kernel_size=3, stride=4,
                            padding=1, bias=True),]
        model1 += [nn.LeakyReLU(True),]
        # 256 x d64
        # model1 += [nn.Conv2d(32, 64, kernel_size=3,
        #                     stride=4, padding=1, bias=True),]
        #model1 += [nn.LeakyReLU(True),]
        # 128 x d64
        model1 += [norm_layer(32),]

        model2 = [nn.Conv2d(32, 64, kernel_size=3,
                            stride=4, padding=1, bias=True),]
        model2 += [nn.LeakyReLU(True),]
        model2 += [nn.Conv2d(64, 64, kernel_size=3,
                             stride=2, padding=1, bias=True),]
        #model2 += [nn.LeakyReLU(True),]
        # 32 x d128
        model2 += [norm_layer(64),]

        # 16 x d128
        model3 = [nn.Conv2d(64, 32, kernel_size=3,
                            stride=2, padding=1, bias=True),]
        model3 += [nn.LeakyReLU(True),]
        model3 += [nn.Conv2d(32, 1, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model3 += [nn.Tanh(),]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)

        # self.model_out = nn.Conv2d(256, 1, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

    def forward(self, input_ir):
        """Caculate grayscale image

        Args:
            input_ir (tensor): tensor of (N,1,W,H) of gracale image

        Returns:
            tensor: Output tensor of (N,) of prob on fake image
        """
        conv1_2 = self.model1(input_ir)
        conv2_2 = self.model2(conv1_2)
        conv3_2 = self.model3(conv2_2)
        out_reg = (torch.mean(conv3_2, dim=(1, 2, 3)) + 1) / 2

        return out_reg


def train(model, d_model, data, learning_rate=0.001, lambda1=1, lambda2=0.2, max_norm=1.0):
    """Train model with generator generate image tensor pair (x,y)

    Args:
        model (nn.Model): PyTorch model
        d_model (nn.Model): PyTorch model for discriminator
        data (Iterable): Iterable object returns (x,y) data pair
        learning_rate (float, optional): Learning rate for training model. Defaults to 0.001.
        lambda1 (int, optional): Weight for Smooth L1 Loss. Defaults to 1.
        lambda2 (float, optional): Weight for discriminator Loss. Defaults to 0.05.

    Returns:
        (float, float): Pair of total loss and discriminator training Loss
    """

    # Set model to training mode
    model.train()
    d_model.train()
    # Init opt and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()
    criterion_d = torch.nn.L1Loss()
    # Return total loss for stat
    total_loss = 0
    count = 0
    # Iterate through data
    for x, y in data:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # Forward
        optimizer.zero_grad()
        # d_model.eval()
        output = model(x)
        # Calculate Loss
        loss = criterion(output, y) * lambda1
        # Caculate Discriminator Loss
        d_out_o = d_model(output)
        d_out_y = d_model(y)
        # - torch.mean(torch.log(1-d_out_o))
        d_loss = torch.mean(criterion_d(d_out_o, d_out_y))
        loss += d_loss * lambda2
        # Calculate stat infomation
        total_loss += loss.item()
        count += 1
        # Step on generator
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # Print state
        d_out_o = torch.mean(d_out_o)
        d_out_y = torch.mean(d_out_y)
        print('\r'+format(count, ' 6'),
              "LOSS="+format(loss, '.3g'),
              "D_LOSS="+format(d_loss, '.3g'),
              "D_OUT_O="+format(d_out_o, '.3g'),
              "D_OUT_Y="+format(d_out_y, '.3g'), end='           ')
    # Clear line
    print('\r                                       ' +
          '                                              ',
          end='')
    return total_loss / count


def train_d(model, d_model, data, d_learning_rate=0.005, max_norm=1.0):
    """Train model with generator generate image tensor pair (x,y)

    Args:
        model (nn.Model): PyTorch model
        d_model (nn.Model): PyTorch model for discriminator
        data (Iterable): Iterable object returns (x,y) data pair
        d_learning_rate (float, optional): Learning rate for discriminator model. Defaults to 0.005.
        lambda1 (int, optional): Weight for Smooth L1 Loss. Defaults to 1.
        lambda2 (float, optional): Weight for discriminator Loss. Defaults to 0.05.

    Returns:
        (float, float): Pair of total loss and discriminator training Loss
    """

    # Set model to training mode
    model.train()
    d_model.train()
    # Init opt and loss
    optimizer_d = torch.optim.SGD(
        d_model.parameters(), lr=d_learning_rate)
    # Return total loss for stat
    total_loss_d = 0
    count = 0
    # Iterate through data
    for x, y in data:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # d_model.eval()
        output = model(x)
        # Calculate stat infomation
        count += 1
        # Train discriminator for invalid input
        optimizer_d.zero_grad()
        d_out_o = d_model(output.detach())
        d_out_y = d_model(y)
        loss_d = - torch.mean(torch.log(d_out_o)) - \
            torch.mean(torch.log(1-d_out_y))
        total_loss_d += loss_d.item()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(d_model.parameters(), max_norm)
        optimizer_d.step()
        # Print state
        d_out_o = torch.mean(d_out_o)
        d_out_y = torch.mean(d_out_y)
        print('\r'+format(count, ' 6'),
              "LOSS="+format(loss_d, '.3g'),
              "D_OUT_O="+format(d_out_o, '.3g'),
              "D_OUT_Y="+format(d_out_y, '.3g'), end='           ')
    # Clear line
    print('\r                                       ' +
          '                                              ',
          end='')
    return total_loss_d / count / 2


def test(model, d_model, data, lambda1=1, lambda2=0.2):
    """Test model with generator generate image tensor  pair (x,y)

    Args:
        model (nn.Model): PyTorch model
        data (Iterable): Iterable object returns (x,y) data pair
        lambda1 (int, optional): Weight for Smooth L1 Loss. Defaults to 1.
        lambda2 (float, optional): Weight for discriminator Loss. Defaults to 0.05.

    Returns:
        float: Average loss on testing set
    """
    # Set model to evaluation mode
    model.eval()
    d_model.eval()
    # Init loss
    criterion = torch.nn.SmoothL1Loss()
    criterion_d = torch.nn.L1Loss()
    total_loss = 0
    count = 0
    for x, y in data:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # Forward
        output = model(x)
        # Calculate Loss
        loss = criterion(output, y) * lambda1
        d_out_o = d_model(output)
        d_out_y = d_model(y)
        # - torch.mean(torch.log(1-d_out_o))
        d_loss = torch.mean(criterion_d(d_out_o, d_out_y))
        loss += d_loss * lambda2
        # Calculate stat infomation
        total_loss += loss.item()
        count += 1
        # Print state
        print('\r[TEST]', format(count, ' 6'), " D_LOSS=",
              format(d_loss, '.3g'), sep='',
              end='                 ')
    # Clear line
    print('\r                                            ' +
          '                                         ', end='')
    return total_loss / count


def adap_train(model, d_model, data, learning_rate=0.001, lambda1=1, lambda2=0.2, max_norm=1.0):
    """Train generator model with image tensor pair (x,y), yaild on single step

    Args:
        model (nn.Model): PyTorch model
        d_model (nn.Model): PyTorch model for discriminator
        data (Iterable): Iterable object returns (x,y) data pair
        learning_rate (float, optional): Learning rate for training model. Defaults to 0.001.
        lambda1 (int, optional): _description_. Defaults to 1.
        lambda2 (float, optional): _description_. Defaults to 0.2.
        max_norm (float, optional): Clip on max_norm. Defaults to 1.0.

    Yields:
        (float, float): Pair of D loss and average training Loss until the step
    """

    # Set model to training mode
    model.train()
    d_model.train()
    # Init opt and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()
    criterion_d = torch.nn.L1Loss()
    # Return total loss for stat
    total_loss = 0
    count = 0
    # Iterate through data
    for x, y in data:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # Forward
        optimizer.zero_grad()
        # d_model.eval()
        output = model(x)
        # Calculate Loss
        loss = criterion(output, y) * lambda1
        # Caculate Discriminator Loss
        d_out_o = d_model(output)
        d_out_y = d_model(y)
        # - torch.mean(torch.log(1-d_out_o))
        d_loss = torch.mean(criterion_d(d_out_o, d_out_y))
        loss += d_loss * lambda2
        # Calculate stat infomation
        total_loss += loss.item()
        count += 1
        # Step on generator
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        # Print state
        d_out_o = torch.mean(d_out_o)
        d_out_y = torch.mean(d_out_y)
        print('\r[G]'+format(count, ' 6'),
              "LOSS="+format(loss, '.3g'),
              "D_LOSS="+format(d_loss, '.3g'),
              "D_OUT_O="+format(d_out_o, '.3g'),
              "D_OUT_Y="+format(d_out_y, '.3g'), end='           ')
        yield d_loss.item(), total_loss / count


def adap_train_d(model, d_model, data, d_learning_rate=0.005, max_norm=1.0):
    """Train discriminator model with generator generate image tensor pair (x,y), yaild on single step

    Args:
        model (nn.Model): PyTorch model
        d_model (nn.Model): PyTorch model for discriminator
        data (Iterable): Iterable object returns (x,y) data pair
        d_learning_rate (float, optional): Learning rate for training model. Defaults to 0.005.
        max_norm (float, optional): Clip on max_norm. Defaults to 1.0.

    Yields:
        (float, float): Pair of current loss and average training Loss until the step
    """

    # Set model to training mode
    model.train()
    d_model.train()
    # Init opt and loss
    optimizer_d = torch.optim.SGD(d_model.parameters(), lr=d_learning_rate)
    # Return total loss for stat
    total_loss_d = 0
    count = 0
    # Iterate through data
    for x, y in data:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # d_model.eval()
        output = model(x)
        # Calculate stat infomation
        count += 1
        # Train discriminator for invalid input
        optimizer_d.zero_grad()
        d_out_o = d_model(output.detach())
        d_out_y = d_model(y)
        loss_d = - torch.mean(torch.log(d_out_o)) - \
            torch.mean(torch.log(1-d_out_y))
        total_loss_d += loss_d.item()
        loss_d.backward()
        torch.nn.utils.clip_grad_norm_(d_model.parameters(), max_norm)
        optimizer_d.step()
        # Print state
        d_out_o = torch.mean(d_out_o)
        d_out_y = torch.mean(d_out_y)
        print('\r[D]'+format(count, ' 6'),
              "LOSS="+format(loss_d, '.3g'),
              "D_OUT_O="+format(d_out_o, '.3g'),
              "D_OUT_Y="+format(d_out_y, '.3g'), end='           ')
        yield loss_d.item(), total_loss_d / count
