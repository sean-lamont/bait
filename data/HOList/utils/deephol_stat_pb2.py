# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: experiments/holist/utilities/deephol_stat.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from experiments.HOList import deephol_pb2 as experiments_dot_holist_dot_deephol__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n/experiments/holist/utilities/deephol_stat.proto\x12\x10\x64\x65\x65pmath_deephol\x1a experiments/holist/deephol.proto\"t\n\x11LogScaleHistogram\x12\x35\n\x01h\x18\x01 \x03(\x0b\x32*.deepmath_deephol.LogScaleHistogram.HEntry\x1a(\n\x06HEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\"\xa4\x03\n\tProofStat\x12\x1e\n\x16num_theorems_attempted\x18\x01 \x01(\x05\x12\x1b\n\x13num_theorems_proved\x18\x02 \x01(\x05\x12#\n\x1bnum_theorems_with_bad_proof\x18\x03 \x01(\x05\x12\x11\n\tnum_nodes\x18\x04 \x01(\x03\x12\x1c\n\x14reduced_node_indices\x18\x05 \x03(\x05\x12\x1b\n\x13\x63losed_node_indices\x18\x06 \x03(\x05\x12\x1f\n\x17time_spent_milliseconds\x18\x07 \x01(\x04\x12\x1b\n\x13theorem_fingerprint\x18\x08 \x01(\x04\x12:\n\ttapp_stat\x18\t \x01(\x0b\x32\'.deepmath_deephol.TacticApplicationStat\x12 \n\x15total_prediction_time\x18\n \x01(\x03:\x01\x30\x12K\n\x1enode_prediction_time_histogram\x18\x0b \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\"\x9a\x07\n\x12ProofAggregateStat\x12!\n\x16num_theorems_attempted\x18\x01 \x01(\x05:\x01\x30\x12\x1e\n\x13num_theorems_proved\x18\x02 \x01(\x05:\x01\x30\x12&\n\x1bnum_theorems_with_bad_proof\x18\x03 \x01(\x05:\x01\x30\x12\x14\n\tnum_nodes\x18\x04 \x01(\x03:\x01\x30\x12\x1c\n\x11num_reduced_nodes\x18\x05 \x01(\x03:\x01\x30\x12\x1b\n\x10num_closed_nodes\x18\x06 \x01(\x03:\x01\x30\x12\"\n\x17time_spent_milliseconds\x18\x07 \x01(\x04:\x01\x30\x12\x41\n\x14proof_time_histogram\x18\t \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12H\n\x1bproof_time_histogram_proved\x18\n \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12H\n\x1bproof_time_histogram_failed\x18\x0b \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12m\n\x1enum_reduced_nodes_distribution\x18\r \x03(\x0b\x32\x45.deepmath_deephol.ProofAggregateStat.NumReducedNodesDistributionEntry\x12 \n\x15total_prediction_time\x18\x0e \x01(\x03:\x01\x30\x12L\n\x1fproof_prediction_time_histogram\x18\x0f \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12K\n\x1enode_prediction_time_histogram\x18\x10 \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12:\n\ttapp_stat\x18\x0c \x01(\x0b\x32\'.deepmath_deephol.TacticApplicationStat\x12!\n\x19proof_closed_after_millis\x18\x11 \x03(\x04\x1a\x42\n NumReducedNodesDistributionEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\"\xb2\x02\n\x0eTacticTimeStat\x12\x15\n\ntotal_time\x18\x01 \x01(\x03:\x01\x30\x12?\n\x12total_distribution\x18\x02 \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12\x41\n\x14success_distribution\x18\x03 \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12\x43\n\x16unchanged_distribution\x18\x04 \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\x12@\n\x13\x66\x61iled_distribution\x18\x05 \x01(\x0b\x32#.deepmath_deephol.LogScaleHistogram\"\x9f\x1c\n\x15TacticApplicationStat\x12g\n\x1atime_spent_per_tapp_result\x18\x01 \x03(\x0b\x32\x43.deepmath_deephol.TacticApplicationStat.TimeSpentPerTappResultEntry\x12^\n\x15time_spent_per_tactic\x18\x02 \x03(\x0b\x32?.deepmath_deephol.TacticApplicationStat.TimeSpentPerTacticEntry\x12{\n$total_tactic_applications_per_tactic\x18\x03 \x03(\x0b\x32M.deepmath_deephol.TacticApplicationStat.TotalTacticApplicationsPerTacticEntry\x12\x85\x01\n)successful_tactic_applications_per_tactic\x18\x04 \x03(\x0b\x32R.deepmath_deephol.TacticApplicationStat.SuccessfulTacticApplicationsPerTacticEntry\x12\x83\x01\n(unchanged_tactic_applications_per_tactic\x18\x05 \x03(\x0b\x32Q.deepmath_deephol.TacticApplicationStat.UnchangedTacticApplicationsPerTacticEntry\x12}\n%failed_tactic_applications_per_tactic\x18\x06 \x03(\x0b\x32N.deepmath_deephol.TacticApplicationStat.FailedTacticApplicationsPerTacticEntry\x12\x7f\n&unknown_tactic_applications_per_tactic\x18\x07 \x03(\x0b\x32O.deepmath_deephol.TacticApplicationStat.UnknownTacticApplicationsPerTacticEntry\x12\x7f\n&closing_tactic_applications_per_tactic\x18\x08 \x03(\x0b\x32O.deepmath_deephol.TacticApplicationStat.ClosingTacticApplicationsPerTacticEntry\x12p\n\x1e\x63losed_applications_per_tactic\x18\x16 \x03(\x0b\x32H.deepmath_deephol.TacticApplicationStat.ClosedApplicationsPerTacticEntry\x12\x34\n\nmeson_stat\x18\t \x01(\x0b\x32 .deepmath_deephol.TacticTimeStat\x12\x36\n\x0crewrite_stat\x18\n \x01(\x0b\x32 .deepmath_deephol.TacticTimeStat\x12\x33\n\tsimp_stat\x18\x0b \x01(\x0b\x32 .deepmath_deephol.TacticTimeStat\x12O\n\rtime_per_rank\x18\x0c \x03(\x0b\x32\x38.deepmath_deephol.TacticApplicationStat.TimePerRankEntry\x12Q\n\x0etotal_per_rank\x18\r \x03(\x0b\x32\x39.deepmath_deephol.TacticApplicationStat.TotalPerRankEntry\x12U\n\x10success_per_rank\x18\x0e \x03(\x0b\x32;.deepmath_deephol.TacticApplicationStat.SuccessPerRankEntry\x12S\n\x0f\x66\x61iled_per_rank\x18\x0f \x03(\x0b\x32:.deepmath_deephol.TacticApplicationStat.FailedPerRankEntry\x12Y\n\x12unchanged_per_rank\x18\x10 \x03(\x0b\x32=.deepmath_deephol.TacticApplicationStat.UnchangedPerRankEntry\x12S\n\x0f\x63losed_per_rank\x18\x18 \x03(\x0b\x32:.deepmath_deephol.TacticApplicationStat.ClosedPerRankEntry\x12Q\n\x0etime_per_score\x18\x11 \x03(\x0b\x32\x39.deepmath_deephol.TacticApplicationStat.TimePerScoreEntry\x12S\n\x0ftotal_per_score\x18\x12 \x03(\x0b\x32:.deepmath_deephol.TacticApplicationStat.TotalPerScoreEntry\x12W\n\x11success_per_score\x18\x13 \x03(\x0b\x32<.deepmath_deephol.TacticApplicationStat.SuccessPerScoreEntry\x12U\n\x10\x66\x61iled_per_score\x18\x14 \x03(\x0b\x32;.deepmath_deephol.TacticApplicationStat.FailedPerScoreEntry\x12[\n\x13unchanged_per_score\x18\x15 \x03(\x0b\x32>.deepmath_deephol.TacticApplicationStat.UnchangedPerScoreEntry\x12U\n\x10\x63losed_per_score\x18\x17 \x03(\x0b\x32;.deepmath_deephol.TacticApplicationStat.ClosedPerScoreEntry\x1a=\n\x1bTimeSpentPerTappResultEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1a\x39\n\x17TimeSpentPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1aG\n%TotalTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aL\n*SuccessfulTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aK\n)UnchangedTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aH\n&FailedTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aI\n\'UnknownTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1aI\n\'ClosingTacticApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x42\n ClosedApplicationsPerTacticEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x32\n\x10TimePerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1a\x33\n\x11TotalPerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x35\n\x13SuccessPerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x34\n\x12\x46\x61iledPerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x37\n\x15UnchangedPerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x34\n\x12\x43losedPerRankEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x33\n\x11TimePerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x03:\x02\x38\x01\x1a\x34\n\x12TotalPerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x36\n\x14SuccessPerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x35\n\x13\x46\x61iledPerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x38\n\x16UnchangedPerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01\x1a\x35\n\x13\x43losedPerScoreEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x02\x38\x01')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'experiments.holist.utilities.deephol_stat_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _LOGSCALEHISTOGRAM_HENTRY._options = None
  _LOGSCALEHISTOGRAM_HENTRY._serialized_options = b'8\001'
  _PROOFAGGREGATESTAT_NUMREDUCEDNODESDISTRIBUTIONENTRY._options = None
  _PROOFAGGREGATESTAT_NUMREDUCEDNODESDISTRIBUTIONENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TIMESPENTPERTAPPRESULTENTRY._options = None
  _TACTICAPPLICATIONSTAT_TIMESPENTPERTAPPRESULTENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TIMESPENTPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_TIMESPENTPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TOTALTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_TOTALTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_SUCCESSFULTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_SUCCESSFULTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_UNCHANGEDTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_UNCHANGEDTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_FAILEDTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_FAILEDTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_UNKNOWNTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_UNKNOWNTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_CLOSINGTACTICAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_CLOSINGTACTICAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_CLOSEDAPPLICATIONSPERTACTICENTRY._options = None
  _TACTICAPPLICATIONSTAT_CLOSEDAPPLICATIONSPERTACTICENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TIMEPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_TIMEPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TOTALPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_TOTALPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_SUCCESSPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_SUCCESSPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_FAILEDPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_FAILEDPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_UNCHANGEDPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_UNCHANGEDPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_CLOSEDPERRANKENTRY._options = None
  _TACTICAPPLICATIONSTAT_CLOSEDPERRANKENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TIMEPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_TIMEPERSCOREENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_TOTALPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_TOTALPERSCOREENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_SUCCESSPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_SUCCESSPERSCOREENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_FAILEDPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_FAILEDPERSCOREENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_UNCHANGEDPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_UNCHANGEDPERSCOREENTRY._serialized_options = b'8\001'
  _TACTICAPPLICATIONSTAT_CLOSEDPERSCOREENTRY._options = None
  _TACTICAPPLICATIONSTAT_CLOSEDPERSCOREENTRY._serialized_options = b'8\001'
  _globals['_LOGSCALEHISTOGRAM']._serialized_start=103
  _globals['_LOGSCALEHISTOGRAM']._serialized_end=219
  _globals['_LOGSCALEHISTOGRAM_HENTRY']._serialized_start=179
  _globals['_LOGSCALEHISTOGRAM_HENTRY']._serialized_end=219
  _globals['_PROOFSTAT']._serialized_start=222
  _globals['_PROOFSTAT']._serialized_end=642
  _globals['_PROOFAGGREGATESTAT']._serialized_start=645
  _globals['_PROOFAGGREGATESTAT']._serialized_end=1567
  _globals['_PROOFAGGREGATESTAT_NUMREDUCEDNODESDISTRIBUTIONENTRY']._serialized_start=1501
  _globals['_PROOFAGGREGATESTAT_NUMREDUCEDNODESDISTRIBUTIONENTRY']._serialized_end=1567
  _globals['_TACTICTIMESTAT']._serialized_start=1570
  _globals['_TACTICTIMESTAT']._serialized_end=1876
  _globals['_TACTICAPPLICATIONSTAT']._serialized_start=1879
  _globals['_TACTICAPPLICATIONSTAT']._serialized_end=5494
  _globals['_TACTICAPPLICATIONSTAT_TIMESPENTPERTAPPRESULTENTRY']._serialized_start=4198
  _globals['_TACTICAPPLICATIONSTAT_TIMESPENTPERTAPPRESULTENTRY']._serialized_end=4259
  _globals['_TACTICAPPLICATIONSTAT_TIMESPENTPERTACTICENTRY']._serialized_start=4261
  _globals['_TACTICAPPLICATIONSTAT_TIMESPENTPERTACTICENTRY']._serialized_end=4318
  _globals['_TACTICAPPLICATIONSTAT_TOTALTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4320
  _globals['_TACTICAPPLICATIONSTAT_TOTALTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4391
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSFULTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4393
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSFULTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4469
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4471
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4546
  _globals['_TACTICAPPLICATIONSTAT_FAILEDTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4548
  _globals['_TACTICAPPLICATIONSTAT_FAILEDTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4620
  _globals['_TACTICAPPLICATIONSTAT_UNKNOWNTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4622
  _globals['_TACTICAPPLICATIONSTAT_UNKNOWNTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4695
  _globals['_TACTICAPPLICATIONSTAT_CLOSINGTACTICAPPLICATIONSPERTACTICENTRY']._serialized_start=4697
  _globals['_TACTICAPPLICATIONSTAT_CLOSINGTACTICAPPLICATIONSPERTACTICENTRY']._serialized_end=4770
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDAPPLICATIONSPERTACTICENTRY']._serialized_start=4772
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDAPPLICATIONSPERTACTICENTRY']._serialized_end=4838
  _globals['_TACTICAPPLICATIONSTAT_TIMEPERRANKENTRY']._serialized_start=4840
  _globals['_TACTICAPPLICATIONSTAT_TIMEPERRANKENTRY']._serialized_end=4890
  _globals['_TACTICAPPLICATIONSTAT_TOTALPERRANKENTRY']._serialized_start=4892
  _globals['_TACTICAPPLICATIONSTAT_TOTALPERRANKENTRY']._serialized_end=4943
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSPERRANKENTRY']._serialized_start=4945
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSPERRANKENTRY']._serialized_end=4998
  _globals['_TACTICAPPLICATIONSTAT_FAILEDPERRANKENTRY']._serialized_start=5000
  _globals['_TACTICAPPLICATIONSTAT_FAILEDPERRANKENTRY']._serialized_end=5052
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDPERRANKENTRY']._serialized_start=5054
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDPERRANKENTRY']._serialized_end=5109
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDPERRANKENTRY']._serialized_start=5111
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDPERRANKENTRY']._serialized_end=5163
  _globals['_TACTICAPPLICATIONSTAT_TIMEPERSCOREENTRY']._serialized_start=5165
  _globals['_TACTICAPPLICATIONSTAT_TIMEPERSCOREENTRY']._serialized_end=5216
  _globals['_TACTICAPPLICATIONSTAT_TOTALPERSCOREENTRY']._serialized_start=5218
  _globals['_TACTICAPPLICATIONSTAT_TOTALPERSCOREENTRY']._serialized_end=5270
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSPERSCOREENTRY']._serialized_start=5272
  _globals['_TACTICAPPLICATIONSTAT_SUCCESSPERSCOREENTRY']._serialized_end=5326
  _globals['_TACTICAPPLICATIONSTAT_FAILEDPERSCOREENTRY']._serialized_start=5328
  _globals['_TACTICAPPLICATIONSTAT_FAILEDPERSCOREENTRY']._serialized_end=5381
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDPERSCOREENTRY']._serialized_start=5383
  _globals['_TACTICAPPLICATIONSTAT_UNCHANGEDPERSCOREENTRY']._serialized_end=5439
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDPERSCOREENTRY']._serialized_start=5441
  _globals['_TACTICAPPLICATIONSTAT_CLOSEDPERSCOREENTRY']._serialized_end=5494
# @@protoc_insertion_point(module_scope)