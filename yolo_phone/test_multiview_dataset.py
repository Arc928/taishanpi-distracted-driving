"""Safety checks for split leakage and review-gated detection-label export."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from core.metrics import categorize, image_status
from export_multiview import review_label
from prepare_targeted_round import pick_negatives
from core.splits import check_disjoint, select_frames, split_subjects


class DatasetSafetyTests(unittest.TestCase):
    def test_subject_split_is_reproducible_and_complete(self):
        subjects = {f'S{i}' for i in range(1,51)}
        a = split_subjects(subjects)
        self.assertEqual(a,split_subjects(reversed(sorted(subjects))))
        self.assertEqual(Counter(a.values()),{'train':35,'val':5,'test':10})

    def test_sampling_uses_numeric_temporal_order(self):
        frames = [{'frame':n} for n in (100,2,1,10,50,20)]
        self.assertEqual([r['frame'] for r in select_frames(frames,3)],[2,20,100])

    def test_different_views_cannot_cross_split(self):
        rows = [dict(source='3mdad',subject='S1',split='train',image_sha256='a'),
                dict(source='3mdad',subject='S1',split='val',image_sha256='b')]
        with self.assertRaisesRegex(ValueError,'Subject'):
            check_disjoint(rows)

    def test_duplicate_pixels_cannot_cross_split(self):
        rows = [dict(source='3mdad',subject='S1',split='train',image_sha256='same'),
                dict(source='3mdad',subject='S2',split='test',image_sha256='same')]
        with self.assertRaisesRegex(ValueError,'Identical'):
            check_disjoint(rows)

    def test_statefarm_eval_driver_is_forbidden_in_training(self):
        with self.assertRaisesRegex(ValueError,'evaluation-driver'):
            check_disjoint([dict(source='statefarm',subject='p016',split='train',image_sha256='a')])

    def review(self, behavior, decision, shapes):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'example.json'
            path.write_text(json.dumps({'imageWidth':640,'imageHeight':480,
                                        'imagePath':'example.jpg','shapes':shapes}))
            return review_label({'id':'example','image':'example.jpg','source_behavior':behavior},decision,path)

    def test_missed_positive_cannot_become_background(self):
        for behavior in ('texting','calling'):
            with self.assertRaises(ValueError):
                self.review(behavior,'background',[])
            with self.assertRaises(ValueError):
                self.review(behavior,'positive',[])

    def test_reviewed_background_can_export_empty_label(self):
        self.assertEqual(self.review('background_candidate','background',[]),'')

    def test_human_can_correct_weak_class_without_changing_class_ids(self):
        shapes = [{'shape_type':'rectangle','label':'texting','points':[[100,100],[200,200]]}]
        self.assertTrue(self.review('calling','positive',shapes).startswith('0 '))

    def test_unknown_class_and_invalid_boxes_rejected(self):
        for label, points in [('phone',[[0,0],[10,10]]),('calling',[[0,0],[650,480]])]:
            with self.assertRaises(ValueError):
                self.review('calling','positive',[{'shape_type':'rectangle','label':label,'points':points}])

    def test_empty_source_positive_is_not_visible_not_background(self):
        box = [(0,[10,10,50,50])]
        self.assertEqual(categorize({'id':'a','source_behavior':'texting'},[],{}),'not_visible')
        self.assertEqual(categorize({'id':'a','source_behavior':'background_candidate'},[],{}),'background')
        self.assertEqual(categorize({'id':'a','source_behavior':'background_candidate'},box,{}),'positive')
        self.assertEqual(categorize({'id':'a','source_behavior':'texting'},box,{'a':'blurred'}),'exclude')

    def test_holdout_hit_requires_class_and_location(self):
        gt = [(0,[100,100,200,200])]
        pred = lambda c,xy: [{'class_id':c,'confidence':.9,'xyxy':xy}]
        self.assertEqual(image_status('positive',gt,pred(0,[105,105,205,205])),'localized_hit')
        self.assertEqual(image_status('positive',gt,pred(0,[300,300,400,400])),'class_hit_poor_box')
        self.assertEqual(image_status('positive',gt,pred(1,[100,100,200,200])),'wrong_class_only')
        self.assertEqual(image_status('positive',gt,[]),'missed')
        self.assertEqual(image_status('background',[],pred(1,[0,0,9,9])),'false_positive')

    def test_hard_negatives_spread_over_groups_before_second_picks(self):
        scores = {'a':[(.9,1,'a1'),(.8,5,'a2'),(.7,40,'a3')],'b':[(.6,1,'b1')],'c':[(.55,1,'c1')]}
        picks = pick_negatives(scores,2,4,15)
        self.assertEqual([p[3] for p in picks],['a1','b1','c1','a3'])
        self.assertEqual(len(pick_negatives(scores,2,2,15)),2)


if __name__ == '__main__':
    unittest.main()
