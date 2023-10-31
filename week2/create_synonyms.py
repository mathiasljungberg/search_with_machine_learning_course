import fasttext
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    general = parser.add_argument_group("general")
    general.add_argument("--threshold", default=0.75,  help="Threshold for synonym similarity")
    args = parser.parse_args()

    threshold = args.threshold #0.75
    print(threshold)
    model_path = "/workspace/datasets/fasttext/title_model.bin"
    top_words_path = "/workspace/datasets/fasttext/top_words.txt"
    output_file = "/workspace/datasets/fasttext/synonyms.csv"

    model = fasttext.load_model(model_path)

    with open(top_words_path, 'r') as f:
        top_words = [x.strip() for x in f]
    
    with open(output_file, 'w') as output:
        for word in top_words:
            nn = model.get_nearest_neighbors(word)
            line = ", ".join([str(x[1]) for x in nn if x[0] > threshold])
            if len(line) > 0:
                print(f'{word}, {line}')
                output.write(f'{word}, {line}\n')


