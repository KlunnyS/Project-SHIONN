"""Explicit startup timing comparison remains guarded and reproducible."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import run_imitation
import run_model_sequence as seq


class StartupSettleArgsTest(unittest.TestCase):
    def test_direct_runner_rejects_invalid_or_unused_override(self):
        for extra in (['--post-preflight-settle','3'],['--mouse-preflight','--post-preflight-settle','-1'],['--mouse-preflight','--post-preflight-settle','nan'],['--mouse-preflight','--post-preflight-settle','inf']):
            with self.subTest(extra=extra),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):run_imitation.parse_args(['--checkpoint','unused.pt']+extra)

    def test_sequence_forwards_only_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'best.pt';p.touch();p.with_suffix('.json').write_text('{}')
            args=seq.parse_args(['--checkpoint',str(p),'--map','dataset_test1','--post-preflight-settle','0.25'])
            job=seq.build_jobs(args.models,args.maps,args.repeats,args.order)[0]
            cmd=seq.runner_command(job,Path(tmp),args)
            self.assertEqual(cmd[cmd.index('--post-preflight-settle')+1],'0.25')
            args.post_preflight_settle=None
            self.assertNotIn('--post-preflight-settle',seq.runner_command(job,Path(tmp),args))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):seq.parse_args(['--checkpoint',str(p),'--map','dataset_test1','--no-mouse-preflight','--post-preflight-settle','3'])
