"""Module to create a report document from SDMetrics analysis data."""
import re
from collections import namedtuple
from os.path import abspath, dirname, join
import datetime
import pathlib

from jinja2 import Environment, FileSystemLoader
import pickle
import zipfile

import sdmetrics.reports
import sdmetrics.reports.single_table


ImagesDict = namedtuple('ImagesDict', 'column_pair_trends column_shapes per_column')


# ASyH.report: RealData, SyntheticData -> sdmetrics.report
# from the report, extract scores and images
# output markdown, attempt transforming to pdf

class Report:
    SPECIAL_CHARS_REGEXP = re.compile(r'[^a-zA-Z0-9.,:;_-]')

    def __init__(self, input_data, synthetic_data, metadata, sdmetrics_report=None):
        if sdmetrics_report is None:
            sdmetrics_report = sdmetrics.reports.single_table.QualityReport()
        self._input_data = input_data
        self._synthetic_data = synthetic_data
        self._metadata = metadata
        self._sdmetrics_report = sdmetrics_report
        self._report_name = None
        self._image_dir = None
        self._files = []

    def generate(self, dataset_name, sd_model_name, details=False):
        self._prepare(dataset_name)
        self._sdmetrics_report.generate(
            self._input_data,
            self._synthetic_data,
            self._metadata
        )
        self._dump(lambda x: self.create_pickled_report(x), f'{self._report_name}.pkl')
        if details:
            self._dump(lambda x: self.create_scores_csv(x), f'{self._report_name}_scores.csv', mode='w')

        images = self._dump_images()
        markdown = self.get_mark_down_report(dataset_name, sd_model_name, images)
        self._dump(lambda x: print(markdown, file=x), f'{self._report_name}.md', mode='w')
        self._try_to_dump_pdf_report(markdown)
        self.create_zip_archive()

    def _prepare(self, dataset_name):
        self._files = []
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self._report_name = f'report-{dataset_name}-{timestamp}'
        self._image_dir = f'pngs-{dataset_name}-{timestamp}'

    def _dump(self, generator, path, mode='wb'):
        with open(path, mode) as fp:
            generator(fp)
        self._files.append(path)

    def create_pickled_report(self, file_like):
        pickle.dump(self._sdmetrics_report, file_like)

    def create_scores_csv(self, file_like):
        detailed_scores = self._sdmetrics_report.get_details(property_name='Column Shapes')
        detailed_scores.to_csv(file_like)

    def _dump_images(self):
        pathlib.Path(self._image_dir).mkdir(exist_ok=False)

        column_pair_trends_image = self.get_stat_image_name('Column Pair Trends')
        self._dump(lambda x: self.create_stats_image('Column Pair Trends', x), column_pair_trends_image)
        column_shapes_image = self.get_stat_image_name('Column Shapes')
        self._dump(lambda x: self.create_stats_image('Column Shapes', x), column_shapes_image)

        per_column_images = []
        for column in self.get_columns():
            image = self.get_per_column_image_name(column)
            self._dump(lambda x: self.create_per_column_image(column, x), image)
            per_column_images.append(image)

        return ImagesDict(
            column_pair_trends=column_pair_trends_image,
            column_shapes=column_shapes_image,
            per_column=per_column_images,
        )

    def get_columns(self):
        columns = self._metadata['columns'].keys()
        if 'primary_key' in self._metadata.keys():
            columns = [
                c
                for c in columns
                if c != self._metadata['primary_key']
            ]
        return [
            c
            for c in columns
            if self._metadata['columns'][c]['sdtype'] != 'id'
        ]

    def get_mark_down_report(self, dataset_name, sd_model_name, images):
        jinja_template = self._get_report_template()
        return jinja_template.render(
            quality_score_percent=100 * self._sdmetrics_report.get_score(),
            dataset=dataset_name,
            sd_model=sd_model_name,
            column_shapes_score_percent=self.get_report_property_as_percent('Column Shapes'),
            column_pair_trends_score_percent=self.get_report_property_as_percent('Column Pair Trends'),
            column_shapes_image=images.column_shapes,
            column_pair_trends_image=images.column_pair_trends,
            per_column_images=images.per_column
        )

    @staticmethod
    def _get_report_template():
        template_path = dirname(abspath(__file__))
        loader = FileSystemLoader(searchpath=template_path)
        env = Environment(loader=loader)
        return env.get_template('report.j2')

    def get_report_property_as_percent(self, property_name):
        props = self._sdmetrics_report.get_properties()
        return 100 * props[props['Property'] == property_name].iloc[0]['Score']

    def create_per_column_image(self, column, file_like):
        fig = sdmetrics.reports.utils.get_column_plot(
            real_data=self._input_data,
            synthetic_data=self._synthetic_data,
            column_name=column,
            metadata=self._metadata
        )
        fig.write_image(file_like, format='png')

    def get_per_column_image_name(self, column):
        clean_column = self._clean_string(column)
        return join(self._image_dir, f"column_plot_{clean_column}.png")

    def _clean_string(self, x):
        return re.sub(self.SPECIAL_CHARS_REGEXP, 'X', x.replace(' ', '_'))

    def create_stats_image(self, property_name, file_like):
        fig = self._sdmetrics_report.get_visualization(property_name=property_name)
        fig.write_image(file_like, format='png')

    def get_stat_image_name(self, property_name):
        cooked_property_name = self._clean_string(property_name.lower())
        return join(self._image_dir, f"{cooked_property_name}.png")

    def _try_to_dump_pdf_report(self, markdown):
        try:
            self._dump(lambda x: self._create_pdf_report(markdown, x), f'{self._report_name}.pdf')
        except Exception as exception:
            print('')
            print(f'Could not produce PDF document: {type(exception)}!')
            print('')
            print('  *****************************************************************')
            print('  *  To be able to produce a PDF version of the Report, you need  *')
            print('  *    - the pypandoc Python module installed with pip,           *')
            print('  *    - the pandoc command line utility, and a                   *')
            print('  *    - LaTeX installation, like TeXLive.                        *')
            print('  *****************************************************************')
            print('')

    @staticmethod
    def _create_pdf_report(markdown, file_like):
        import pypandoc
        file_like.write(
            pypandoc.convert_text(markdown, 'pdf', format='md')
        )

    def create_zip_archive(self):
        with zipfile.ZipFile(f'{self._report_name}.zip', 'w') as archive:
            for path in self._files:
                archive.write(path)
