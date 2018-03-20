import os

def dict_from_textfile(in_file, delim=','):
    d = {}
    with open(in_file) as f:
        keys = []
        values = []
        for line in f:
            try:
                (key, val) = line.split('\n')[0].split(delim)
                if(key in d):
                    d[key].append(val)
                else:
                    d[key] = [val]

                keys.append(key)
                values.append(val)
            except:
                pass

    return d, keys, values

# break a list into subgroups
def chunker(seq, size):
    return (seq[pos:pos + size] for pos in xrange(0, len(seq), size))


def list_from_textfile(in_file):
    """ Return a list from file with no separator """

    results = []

    with open(in_file, 'r') as fhandle:
        for line in fhandle:
            line_contents = line.strip().split()

            results.append(line_contents[0])
    return results


def list_from_file(fname, separator):
    """ Return a list from file with arbitrary separator """

    total_list = []
    with open(fname, 'r') as fhandle:
        for line in fhandle:
            if not separator:
                line_contents = line.split()
                total_list.append(line_contents[0])
            else:
                line_contents = line.split(separator)
                total_list.append(line_contents[0])

    return total_list

def print_dictionary(dictionary = None):

    for k,v in dictionary.iteritems():
        print(k, v)


def remove_and_create_dir(path):
    """ System call to rm -rf and then re-create a dir """

    dir = os.path.dirname(path)
    print('attempting to delete ', dir, ' path ', path)
    if os.path.exists(path):
        os.system("rm -rf " + path)
    os.system("mkdir -p " + path)
