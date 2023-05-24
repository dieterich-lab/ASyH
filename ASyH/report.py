"""Module to create a report document from SDMetrics analysis data."""

import datetime
import pathlib

from jinja2 import Environment, BaseLoader
import pickle
import zipfile

import sdmetrics.reports
import sdmetrics.reports.single_table


# ASyH.report: RealData, SyntheticData -> sdmetrics.report
# from the report, extract scores and images
# output markdown, attempt transforming to pdf

class Report:

    _TEMPLATE = """# ASyH/SDMetrics Report for dataset {{ dataset }}

## Best model

**{{ sd_model }}**

**QualityScore: {{ '%3.2f' % quality_score|float }}%**

|      Column Shapes          |        Column Pair Trends       |
| --------------------------- | ------------------------------- |
| {{ '%3.2f' % column_shapes_score|float }} % | {{ '%3.2f' % column_pair_trends_score|float }} % |

## Distribution and Correlation Similarities

![Column distribution similarity comparison.]({{ column_shapes }})

![Column pair trends and column correlation comparison.]({{ column_pair_trends }})

## Per-Column comparisons:

{% for image in images %}
![]({{ image }})
{% endfor %}
"""

    def __init__(self, input_data, synthetic_data, metadata):
        self._input_data = input_data
        self._synthetic_data = synthetic_data
        self._metadata = metadata
        self._sdmetrics_report = sdmetrics.reports.single_table.QualityReport()

    def generate(self, dataset_name, sd_model_name, details=False):
        self._sdmetrics_report.generate(self._input_data,
                                        self._synthetic_data,
                                        self._metadata)

        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        name_time = f'{dataset_name}-{timestamp}'
        filename_stump = f'./report-{name_time}'

        files = []

        # pickling the SDMetrics report object
        pickle_file = f'{filename_stump}.pkl'
        with open(pickle_file, 'wb') as report_pickle:
            pickle.dump(self._sdmetrics_report, report_pickle)

        files.extend([pickle_file])

        if details:
            detailed_scores_file = f'{filename_stump}_scores.csv'
            detailed_scores = self._sdmetrics_report.get_details(property_name='Column Shapes')
            detailed_scores.to_csv(detailed_scores_file)
            files.extend([detailed_scores_file])

        images = self._create_images(f'./pngs-{name_time}')
        files.extend(images)

        markdown = self._create_report_markdown(dataset_name,
                                                sd_model_name,
                                                images)
        markdown_file = f'{filename_stump}.md'
        with open(markdown_file, 'w', encoding='utf-8') as md_file:
            md_file.write(markdown)

        files.extend([markdown_file])

        try:
            import pypandoc
            pdf = f'{filename_stump}.pdf'
            pypandoc.convert_text(markdown,
                                  'pdf',
                                  format='md',
                                  outputfile=pdf)
            files.extend([pdf])
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

        zip_file = f'{filename_stump}.zip'
        with zipfile.ZipFile(f'{filename_stump}.zip', 'w') as archive:
            for file in files:
                archive.write(file)

        return zip_file

    def _create_images(self, image_dir):
        image_list = []
        pathlib.Path(image_dir).mkdir(exist_ok=False)

        image_list.append(self._stats_fig('Column Pair Trends', image_dir))
        image_list.append(self._stats_fig('Column Shapes', image_dir))

        column_names = self._metadata['columns'].keys()
        if 'primary_key' in self._metadata.keys():
            column_names = [k for k in column_names
                            if k != self._metadata['primary_key']]
        columns = [k for k in column_names
                   if self._metadata['columns'][k]['sdtype'] != 'id']

        for column in columns:
            image_list.append(self._column_plot(column, image_dir))

        return image_list

    def _create_report_markdown(self, dataset_name, sd_model_name, image_list):
        quality_score = self._sdmetrics_report.get_score() * 100
        props = self._sdmetrics_report.get_properties()
        column_shapes_score = \
            props[props['Property'] == 'Column Shapes'].iloc[0]['Score'] * 100
        column_pair_trends_score = \
            props[props['Property'] == 'Column Pair Trends'].iloc[0]['Score'] * 100

        jinja_env = Environment()
        jinja_template = jinja_env.from_string(self._TEMPLATE)

        column_pair_trends = image_list[0]
        column_shapes = image_list[1]

        return jinja_template.render(loader=BaseLoader(),
                                     quality_score=quality_score,
                                     dataset=dataset_name,
                                     sd_model=sd_model_name,
                                     column_shapes_score=column_shapes_score,
                                     column_pair_trends_score=column_pair_trends_score,
                                     column_shapes=column_shapes,
                                     column_pair_trends=column_pair_trends,
                                     images=image_list[2:])

    def _column_plot(self, column, output_dir):
        fig = sdmetrics.reports.utils.get_column_plot(real_data=self._input_data,
                                                      synthetic_data=self._synthetic_data,
                                                      column_name=column,
                                                      metadata=self._metadata)
        outputfile = f'{output_dir}/column_plot_{column}.png'
        fig.write_image(outputfile, format='png')
        return outputfile

    def _stats_fig(self, property_name, output_dir):
        fig = self._sdmetrics_report.get_visualization(property_name=property_name)
        file_name = property_name.lower().replace(' ', '_')
        outputfile = f'{output_dir}/{file_name}.png'
        fig.write_image(outputfile, format='png')
        return outputfile
