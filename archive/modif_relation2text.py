def replace_dots_in_relations(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in lines:
            parts = line.split("\t")
            if len(parts) > 1:
                parts[1] = parts[1].replace('.', ' is related to ')
                parts[2:] = [part.replace('.', ' is related to ') for part in parts[2:]]
                new_line = "\t".join(parts)
            else:
                new_line = line 
            outfile.write(new_line)

input_file = 'data/FB15kET_sample_2hop/relation2text.txt'
output_file = 'newrelation2text.txt' 

replace_dots_in_relations(input_file, output_file)
