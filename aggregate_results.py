from attack_utils import *
import argparse
import os
'''
There are two APIs for aggregating results:
1. File API: Specify a text file of paths to .pt files to load statistics from. This is the most general and can be used for any use case.

Text file:
Title\n
path1.pt,label1\n
path2.pt,label2\n

Title and labels are optional, but hold precedence over command line args

2. Folder API: Specify a folder and all the .pt files in that folder will be loaded, providing some automation.
You can specify the title through an arg.
'''

def unique_part(files):
    '''
    Very very lazy and vulnerable to breaking code for finding the unique parts of a set of strings
    '''
    for start_index in range(min(len(file) for file in files)):
        if len(set(file[start_index] for file in files))!=1:
            break

    for end_index in range(min(len(file) for file in files)):
        if len(set(file[-(end_index+1)] for file in files))!=1:
            break

    return [file[start_index:-(end_index+1)] for file in files]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', action="store", type=str, required=False, help='Path to text file of format MyTitle\npath1.pt,label1\n...')
    parser.add_argument('--folder', action="store", type=str, required=False, help='Path to folder')
    parser.add_argument('--title', action="store", type=str, required=False, help='Title for the plot')
    parser.add_argument('--log', action="store_true", required=False, help="Log plot")
    args = parser.parse_args()

    if not ((args.file is None) ^ (args.folder is None)):
        raise ValueError(f"Exactly one of the file API (got {args.file}) and folder API (got {args.folder}) must be used.")

    if args.file is not None:
        files = []
        labels = []
        title = None
        with open(args.file,"r") as f:
            for i,line in enumerate(f):
                if i==0 and line[-3:]!=".pt" and line.rfind('.pt,')==-1:
                    title = line.strip()
                else:
                    sep_idx = line.rfind('.pt,')
                    if sep_idx==-1:
                        files.append(line.strip())
                        labels.append(None)
                    else:
                        files.append(line[:sep_idx+3].strip())
                        labels.append(line[sep_idx+4:].strip())
        if title is None:
            title = args.title if args.title is not None else args.file
        
        labels = [(label if label is not None else file) for file,label in zip(files,labels)]
        plot_ROC_files(files, title, labels=labels, save_name=title+("_log" if args.log else "")+".png", log_scale=args.log)

    if args.folder is not None:
        files = []
        for file in os.listdir(args.folder):
            if file[-3:]==".pt":
                files.append(os.path.join(args.folder,file))
        title = args.title if args.title is not None else args.folder
        plot_ROC_files(files, title, labels=unique_part(files), save_name=title+("_log" if args.log else "")+".png", log_scale=args.log)


if __name__ == "__main__":
    main()