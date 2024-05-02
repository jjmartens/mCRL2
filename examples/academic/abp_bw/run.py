#!/usr/bin/env python3

import subprocess

subprocess.run(['mcrl22lps', '-v', 'abp_bw.mcrl2', 'abp_bw.lps'], shell=True, check=True)
subprocess.run(['lps2pbes', '-v', '-f', 'nodeadlock.mcf', 'abp_bw.lps', 'abp_bw.nodeadlock.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.nodeadlock.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'infinitely_often_enabled_then_infinitely_often_taken.mcf', 'abp_bw.lps', 'abp_bw.infinitely_often_enabled_then_infinitely_often_taken.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.infinitely_often_enabled_then_infinitely_often_taken.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'infinitely_often_lost.mcf', 'abp_bw.lps', 'abp_bw.infinitely_often_lost.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.infinitely_often_lost.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'infinitely_often_receive_d1.mcf', 'abp_bw.lps', 'abp_bw.infinitely_often_receive_d1.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.infinitely_often_receive_d1.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'infinitely_often_receive_for_all_d.mcf', 'abp_bw.lps', 'abp_bw.infinitely_often_receive_for_all_d.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.infinitely_often_receive_for_all_d.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'read_then_eventually_send.mcf', 'abp_bw.lps', 'abp_bw.read_then_eventually_send.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.read_then_eventually_send.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'read_then_eventually_send_if_fair.mcf', 'abp_bw.lps', 'abp_bw.read_then_eventually_send_if_fair.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.read_then_eventually_send_if_fair.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'no_generation_of_messages.mcf', 'abp_bw.lps', 'abp_bw.no_generation_of_messages.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.no_generation_of_messages.pbes'], shell=True, check=True)

subprocess.run(['lps2pbes', '-v', '-f', 'no_duplication_of_messages.mcf', 'abp_bw.lps', 'abp_bw.no_duplication_of_messages.pbes'], shell=True, check=True)
subprocess.run(['pbes2bool', '-v', 'abp_bw.no_duplication_of_messages.pbes'], shell=True, check=True)
