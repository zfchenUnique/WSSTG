import torch

def build_opt(args, model):
    if args.optimizer.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif args.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
            lr=args.lr,
            momentum=args.momentum, weight_decay=args.decay)
    else:
        raise Exception()
    return optimizer
