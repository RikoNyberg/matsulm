####################################################################################
# TRAIN
####################################################################################
def train_lstm_model(train_data, valid_data, test_data, params, verbose_logging=True):

    def train_model():
        for epoch in range(params['num_epochs']):
            print('#'*10, f'Epoch [{epoch+1}/{params["num_epochs"]}]', '#'*10)
            
            # learning rate decay
            if params.get('lr_decay') and params.get('lr_decay') != 1:
                new_lr = params['lr'] * (params['lr_decay'] ** max(epoch + 1 - params['lr_decay_start'], 0.0))
                print('Learning rate: {:.4f}'.format(new_lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr
            
            train_epoch_loss = predict(train_data, train=True)
            valid_epoch_loss = predict(valid_data, train=False)

            if verbose_logging:
                print('-'*10, f'End of Epoch {epoch+1}', '-'*10)
                print('Train Loss: {:.4f}, Train Perplexity: {:5.2f}'
                    .format(train_epoch_loss, np.exp(train_epoch_loss)))
                print('Valid Loss: {:.4f}, Valid Perplexity: {:5.2f}'
                    .format(valid_epoch_loss, np.exp(valid_epoch_loss)))
                print('-'*40)
        
        test_epoch_loss = predict(test_data, train=False)            
        print('-'*10, f'Test set results', '-'*10)
        print('Test Loss: {:.4f}, Test Perplexity: {:5.2f}'
                .format(test_epoch_loss, np.exp(test_epoch_loss)))
        
        return True

    def predict(data, train=False):
        if train:
            model.train()
        else:
            model.eval()
        
        # Set initial hidden and cell states
        states = (
            torch.zeros(
                params['model']['num_layers'],# * (2 if params['model']['bidirectional'] else 1), 
                params['batch_size'], 
                params['model']['hidden_size'],
            ).to(device),
            torch.zeros(
                params['model']['num_layers'],# * (2 if params['model']['bidirectional'] else 1), 
                params['batch_size'], 
                params['model']['hidden_size'],
            ).to(device)
        )
        
        losses = []
        for i in range(0, data.size(1) - params['seq_length'], params['seq_length']):
            # Get mini-batch inputs and targets
            inputs = data[:, i:i+params['seq_length']].to(device)
            targets = data[:, (i+1):(i+1)+params['seq_length']].to(device)
            
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            # https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/4
            states = detach(states)

            # Forward pass
            outputs, states = model(inputs, states)
            loss = criterion(outputs, targets.reshape(-1)) # in here the targets.reshape(-1) is the same as the .t() transpose in the batchify
            losses.append(loss.item())

            if train:
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), params['clip_norm'])
                optimizer.step()

            step = (i+1) // params['seq_length']
            if step % params['log_interval'] == 0 and i != 0 and verbose_logging:
                loss_mean = sum(losses[-params['log_interval']:]) / params['log_interval']
                print('Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                    .format(step, data.size(1) // params['seq_length'], loss_mean, np.exp(loss_mean)))
        
        loss_mean = sum(losses) / len(losses)
        return loss_mean


    device = torch.device('cuda' if params['cuda'] and torch.cuda.is_available() else 'cpu')

    print(params["model"])
    
    if params.get('seed'):
        torch.manual_seed(params['seed'])
    model = RNNLM(**params["model"]).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
    else:
        raise ValueError('Missing optimizer parameter')
        
    return model, train_model()
