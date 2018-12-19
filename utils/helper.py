import os

### IO


def check_dir(dir, verbose=True):
    if not os.path.exists(dir):
        if verbose:
            print "Directory %s do not exist; creating..." % dir
        os.makedirs(dir)
        
        
def print_config(config):
    info = "Running with the following configs:\n"
    for k,v in config.items():
        info += "\t%s : %s\n" % (k, str(v))
    print "\n" + info + "\n"
    return