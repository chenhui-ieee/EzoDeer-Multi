"""Validate task routing and overwrite guards without starting a GPU job."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ezodeer_training.cli import main

class CommandTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'dataset';self.output=self.root/'experiment'
        for split in ['train','val','test']:(self.data/split).mkdir(parents=True)

    def invoke(self,mode,task='detection',extra=()):
        argv=[mode+'.py','--task',task,'--data-root',str(self.data),'--output',str(self.output),*extra]
        with patch('sys.argv',argv),patch('subprocess.run') as run:
            with contextlib.redirect_stdout(io.StringIO()):main(mode)
            return [call.args[0] for call in run.call_args_list]

    def test_detection_prepare_then_train(self):
        commands=self.invoke('train')
        self.assertEqual([c[2] for c in commands],['ezodeer_training.detection.prepare_data','ezodeer_training.detection.train'])

    def test_segmentation_prepare_then_train(self):
        commands=self.invoke('train','segmentation')
        self.assertEqual([c[2] for c in commands],['ezodeer_training.segmentation.prepare_data','ezodeer_training.segmentation.train'])

    def test_skip_preparation(self):
        commands=self.invoke('train',extra=['--skip-prepare'])
        self.assertEqual(len(commands),1);self.assertTrue(commands[0][2].endswith('.train'))

    def test_existing_run_preserved(self):
        self.output.mkdir();(self.output/'training_completed.json').write_text('{}')
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:self.invoke('train')
        self.assertEqual(error.exception.code,2)

    def test_test_requires_completed_run(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as error:self.invoke('test')
        self.assertEqual(error.exception.code,2)

    def test_evaluate_then_report(self):
        self.output.mkdir();(self.output/'training_completed.json').write_text('{}')
        commands=self.invoke('test','segmentation')
        self.assertEqual([c[2] for c in commands],['ezodeer_training.segmentation.evaluate','ezodeer_training.segmentation.make_report'])

    def test_skip_report(self):
        self.output.mkdir();(self.output/'training_completed.json').write_text('{}')
        commands=self.invoke('test',extra=['--skip-report'])
        self.assertEqual(len(commands),1);self.assertTrue(commands[0][2].endswith('.evaluate'))

if __name__=='__main__':unittest.main()
