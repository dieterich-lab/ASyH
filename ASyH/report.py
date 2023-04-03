"""Module to create a report document from SDMetrics analysis data."""

import datetime
import pathlib

from jinja2 import Environment, BaseLoader
import pickle

import sdmetrics.reports
import sdmetrics.reports.single_table


# ASyH.report: RealData, SyntheticData -> sdmetrics.report
# from the report, extract scores and images
# output markdown, attempt transforming to pdf


def report(dataset_name,
           sd_model_name,
           input_data,
           synthetic_data,
           metadata):

    sdmetrics_report = sdmetrics.reports.single_table.QualityReport()
    sdmetrics_report.generate(input_data, synthetic_data, metadata)

    quality_score = sdmetrics_report.get_score() * 100
    props = sdmetrics_report.get_properties()
    column_shapes_score = \
        props[props['Property'] == 'Column Shapes'].iloc[0]['Score'] * 100
    column_pair_trends_score = \
        props[props['Property'] == 'Column Pair Trends'].iloc[0]['Score'] * 100

    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    image_dir = f'./pngs-{dataset_name}-{timestamp}'
    report_document = f'./report-{dataset_name}-{timestamp}.md'

    # save the report as a pkl
    pickled_report = pathlib.Path(report_document).with_suffix('.pkl')
    with open(pickled_report, 'wb') as report_pickle:
        pickle.dump(sdmetrics_report, report_pickle)

    # create visualizations of the distributions
    images = create_images(sdmetrics_report,
                           input_data,
                           synthetic_data,
                           metadata,
                           image_dir)

    # create a report in markup format to be converted to pdf:
    documents = create_report_documents(report_document,
                                        dataset_name,
                                        sd_model_name,
                                        quality_score,
                                        column_shapes_score,
                                        column_pair_trends_score,
                                        images)

    # zip the whole thing to be able to transfer the aggregated comparative
    # data.  Still have to make sure the report pickle does not contain the
    # real data!  This is why whe do not include it here.
    import zipfile
    zipfile_name = pathlib.Path(report_document).with_suffix('.zip')
    with zipfile.ZipFile(zipfile_name, 'w') as archive:
        # archive.write(pickled_report)
        for image in images:
            archive.write(image)
        for document in documents:
            archive.write(document)


def create_images(report,
                  input_data,
                  synthetic_data,
                  table_metadata,
                  output_directory):

    image_list = []
    pathlib.Path(output_directory).mkdir(exist_ok=False)

    fig = report.get_visualization(property_name='Column Pair Trends')
    outputfile = f'{output_directory}/column_pair_trends.png'
    fig.write_image(outputfile, format='png')
    image_list.append(outputfile)

    fig = report.get_visualization(property_name='Column Shapes')
    outputfile = f'{output_directory}/column_shapes.png'
    fig.write_image(outputfile, format='png')
    image_list.append(outputfile)

    columns = table_metadata['columns'].keys()
    if 'primary_key' in table_metadata.keys():
        columns = [k for k in columns
                   if k != table_metadata['primary_key']]
    columns = [k for k in columns
               if columns[k]['sdtype'] != 'id']

    for column in columns:
        fig = sdmetrics.reports.utils.get_column_plot(
            real_data=input_data,
            synthetic_data=synthetic_data,
            column_name=column,
            metadata=table_metadata)
        outputfile = f'{output_directory}/column_plot_{column}.png'
        fig.write_image(outputfile, format='png')
        image_list.append(outputfile)

    return image_list


def create_report_documents(report_document,
                            dataset_name,
                            sd_model_name,
                            quality_score,
                            column_shapes_score,
                            column_pair_trends_score,
                            image_list):
    # the output list:
    documents = []

    template = """# ASyH/SDMetrics Report for dataset {{ dataset }}

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
    jinja_env = Environment()
    jinja_template = jinja_env.from_string(template)

    column_pair_trends = image_list[0]
    column_shapes = image_list[1]

    out = jinja_template.render(loader=BaseLoader(),
                                quality_score=quality_score,
                                dataset=dataset_name,
                                sd_model=sd_model_name,
                                column_shapes_score=column_shapes_score,
                                column_pair_trends_score=column_pair_trends_score,
                                column_shapes=column_shapes,
                                column_pair_trends=column_pair_trends,
                                images=image_list[2:])

    with open(report_document, 'w') as md_file:
        md_file.write(out)

    documents.append(report_document)

    try:
        import pypandoc
        pdf_document = pathlib.Path(report_document).with_suffix('pdf')
        pypandoc.convert_text(out,
                              'pdf',
                              format='md',
                              outputfile=pdf_document)
        documents.append(pdf_document)
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

    return documents
