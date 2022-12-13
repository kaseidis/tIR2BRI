
import torch

def train(model, generator, learning_rate=0.05):
    """Train model with generator generate image pair (x,y)"""
    # Set model to training mode
    model.train()
    # Init opt and loss
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()
    # Return total loss for stat
    total_loss = 0
    count = 0
    # Iterate through data
    for x, y in generator:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # Forward
        output = model(x)
        # Calculate Loss
        loss = criterion(output, y)
        # Calculate stat infomation
        total_loss += loss.item()
        count += 1
        # Step on SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print state
        print('\r', count, sep='', end='')
    return total_loss / count

def test(model, generator):
    """Test model with generator generate image tensor pair (x,y)"""
    # Set model to evaluation mode
    model.eval()
    # Init loss
    criterion = torch.nn.SmoothL1Loss()
    total_loss = 0
    count = 0
    for x, y in generator:
        # If GPU avalible, load data in GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            x = x.cuda()
            y = y.cuda()
        # Forward
        output = model(x)
        # Calculate Loss
        loss = criterion(output, y)
        # Calculate stat infomation
        total_loss += loss.item()
        count += 1
        # Print state
        print('\r[Test]', count, sep='', end='')
    return total_loss / count
