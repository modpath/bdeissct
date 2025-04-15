import os
import io
import gzip
import lzma


def get_write_handle(filepath):
    ext = os.path.splitext(filepath)[1]
    if '.gz' == ext:
        return gzip.open(filepath, 'wb')
    if '.xz' == ext:
        return lzma.open(filepath, 'wb')
    return open(filepath, 'w')


def get_read_handle(filepath):
    ext = os.path.splitext(filepath)[1]
    if '.gz' == ext:
        return gzip.open(filepath, 'rt')
    if '.xz' == ext:
        return lzma.open(filepath, 'rt')
    return open(filepath, 'r')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize errors.")
    parser.add_argument('--input', nargs='*', type=str, help="xzipped tables to merge")
    parser.add_argument('--output', type=str, help="merged xzipped table")
    params = parser.parse_args()

    path, ext = os.path.splitext(params.output)
    temp_path = f'{path}.temp{ext}'
    print(temp_path)
    i = 0
    need_header = True
    with get_write_handle(temp_path) as f:
        is_text = isinstance(f, io.TextIOBase)
        for path in params.input:
            with get_read_handle(path) as in_f:
                is_header = True
                for line in in_f:
                    if is_header:
                        if need_header:
                            f.write(line if is_text else line.encode())
                            need_header = False
                        is_header = False
                    else:
                        f.write(line if is_text else line.encode())
                        i += 1
                        if 999 == (i % 1000):
                            print(f'saved {(i + 1):10.0f} trees/forests...')

    os.rename(temp_path, params.output)