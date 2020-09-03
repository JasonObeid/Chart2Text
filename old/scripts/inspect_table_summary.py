import sys
import argparse
import random

NUM_PLAYERS = 13
bs_keys = ["PLAYER_NAME","POS", "MIN", "PTS",
     "FGM", "FGA", "FG_PCT", "FG3M", "FG3A",
     "FG3_PCT", "FTM", "FTA", "FT_PCT", "OREB",
     "DREB", "REB", "AST", "TO", "STL", "BLK",
     "PF", "FIRST_NAME", "SECOND_NAME"]

ls_keys = ["P_QTR1", "P_QTR2", "P_QTR3", "P_QTR4",
    "PTS", "FG_PCT", "FG3_PCT", "FT_PCT", "REB",
    "AST", "TOV", "WINS", "LOSSES", "CITY", "NAME"]

def print_table(table_content, table_marks=None):
    # players + teams + weekday
    assert len(table_content) == (NUM_PLAYERS * 2) * len(bs_keys) + 2 * len(ls_keys) + 1

    bs_head = '{0:15.15}\t{1}\t{2:20.20}{3:20.20}'.format(bs_keys[0], '\t'.join(bs_keys[1:-2]), bs_keys[-2], bs_keys[-1])
    print('=' * len(bs_head))
    print(bs_head)
    for player_idx in range(NUM_PLAYERS * 2):
        player_stats = table_content[player_idx * len(bs_keys) : (player_idx+1) * len(bs_keys)]
        player_stats = [each.split('|')[2] for each in player_stats]
        if table_marks is not None:
            player_marks = table_marks[player_idx * len(bs_keys) : (player_idx+1) * len(bs_keys)]
            player_stats = [mark+stat+mark for stat,mark in zip(player_stats, player_marks)]
        output_text = '{0:15.15}\t{1}\t{2:20.20}{3:20.20}'.format(player_stats[0], '\t'.join(player_stats[1:-2]), player_stats[-2], player_stats[-1])
        print(output_text)
    print('=' * len(bs_head))
    
    ls_head = '{0:10.10}\t{1}\t{2:20.20}{3:20.20}'.format('Home/Away', '\t'.join(ls_keys[:-2]), ls_keys[-2], ls_keys[-1])
    print('=' * len(ls_head))
    print(ls_head)
    home_stats = table_content[2*NUM_PLAYERS*len(bs_keys) : (NUM_PLAYERS * 2) * len(bs_keys) + len(ls_keys)]
    home_stats = [each.split('|')[2] for each in home_stats]
    if table_marks is not None:
        home_marks = table_marks[2*NUM_PLAYERS*len(bs_keys) : (NUM_PLAYERS * 2) * len(bs_keys) + len(ls_keys)]
        home_stats = [mark+stat+mark for stat,mark in zip(home_stats, home_marks)]
    output_home = '{0:10.10}\t{1}\t{2:20.20}{3:20.20}'.format("Home", '\t'.join(home_stats[:-2]), home_stats[-2], home_stats[-1])
    print(output_home)
    vis_stats = table_content[(NUM_PLAYERS * 2) * len(bs_keys) + len(ls_keys) :-1]
    vis_stats = [each.split('|')[2] for each in vis_stats]
    if table_marks is not None:
        vis_marks = table_marks[(NUM_PLAYERS * 2) * len(bs_keys) + len(ls_keys) :-1]
        vis_stats = [mark+stat+mark for stat,mark in zip(vis_stats, vis_marks)]
    output_vis = '{0:10.10}\t{1}\t{2:20.20}{3:20.20}'.format("Away", '\t'.join(vis_stats[:-2]), vis_stats[-2], vis_stats[-1])
    print(output_vis)
    print('-' * len(ls_head))
    print("Game dataOld:", table_content[-1].split('|')[2])
    print('-' * len(ls_head))

if __name__ == '__main__':
    readme = """
    """
    parser = argparse.ArgumentParser(description=readme)
    parser.add_argument("-t", '--table', dest = 'table', help = "Table dataOld")
    parser.add_argument('--table_label', dest = 'table_label', help = "Table label")
    parser.add_argument("-s", '--summary', dest = 'summary', help = "Summary")
    parser.add_argument('--summary_label', dest = 'summary_label', help = "Summary label")
    parser.add_argument("-i", '--index', dest = 'index', type=int, help = "Example index")
    args = parser.parse_args()

    if args.index is None:
        data_size = 0
        if args.summary is not None:
            for _ in open(args.summary, 'r'):
                data_size += 1
        else:
            for _ in open(args.table, 'r'):
                data_size += 1
        example_idx = random.randint(0, data_size-1)
    else:
        example_idx= args.index

    if args.table is not None:
        table_data = []
        if args.table_label is None:
            for table_line in open(args.table, 'r'):
                table_data.append(table_line)
                
            gtable = table_data[example_idx]
            print("Example Index:", example_idx)
            table_content = gtable.split()
            print_table(table_content)
        else:
            table_inf = open(args.table, 'r')
            table_label_inf = open(args.table_label, 'r')
            for table_line, label_line in zip(table_inf, table_label_inf):
                table_data.append((table_line, label_line))

            table_line, label_line = table_data[example_idx]
            table_content = table_line.split()
            label_content = label_line.split()
            assert len(table_content) == len(label_content)
            table_marks = [' '] * len(table_content)
            for idx, l in enumerate(label_content):
                if l != '0':
                    table_marks[idx] = '*'

            print("Example Index:", example_idx)
            print_table(table_content, table_marks)

    if args.summary is not None:
        summary_data = []
        if args.summary_label is None:
            for summary_line in open(args.summary, 'r'):
                summary_data.append(summary_line)
            summary = summary_data[example_idx]
            print("Example Index:", example_idx)
            print(summary)
        else:
            summary_inf = open(args.summary, 'r')
            summary_label_inf = open(args.summary_label, 'r')
            for summary_line, label_line in zip(summary_inf, summary_label_inf):
                summary_data.append((summary_line, label_line))
            summary_line, label_line = summary_data[example_idx]
            words = summary_line.strip().split()
            labels = label_line.strip().split()
            assert len(words) == len(labels)

            output_line = []
            for w, l in zip(words, labels):
                if 0 == int(l):
                    output_line.append(w)
                else:
                    output_line.append('[{}]'.format(w))
            print("Example Index:", example_idx)
            print(' '.join(output_line))
                
