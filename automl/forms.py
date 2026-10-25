from django import forms
from django.conf import settings

from .models import Dataset
from .services import training


class DatasetUploadForm(forms.Form):
    name = forms.CharField(max_length=120)
    file = forms.FileField()

    def clean_file(self):
        f = self.cleaned_data["file"]
        if not f.name.lower().endswith(".csv"):
            raise forms.ValidationError("Only .csv files are supported.")
        if f.size > settings.MAX_UPLOAD_BYTES:
            raise forms.ValidationError(f"File exceeds the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        return f


class TargetForm(forms.Form):
    target_column = forms.CharField(max_length=200)
    problem_type = forms.ChoiceField(choices=[("classification", "Classification"), ("regression", "Regression")])


class TrainForm(forms.Form):
    dataset = forms.ModelChoiceField(queryset=Dataset.objects.none())
    target_column = forms.CharField(max_length=200)
    problem_type = forms.ChoiceField(choices=[("classification", "Classification"), ("regression", "Regression")])
    algorithms = forms.MultipleChoiceField(choices=[(v, l) for v, l, _ in training.algorithm_choices()],
                                           widget=forms.CheckboxSelectMultiple)
    random_state = forms.IntegerField(initial=42, min_value=0, max_value=2**31 - 1)

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.fields["dataset"].queryset = Dataset.objects.all()

    def clean(self):
        data = super().clean()
        ds, target, pt = data.get("dataset"), data.get("target_column"), data.get("problem_type")
        if ds and target and target not in [f["name"] for f in ds.profile.get("features", [])]:
            self.add_error("target_column", "This column does not exist in the selected dataset.")
        algos = data.get("algorithms") or []
        if pt and any(not a.startswith(pt + ":") for a in algos):
            self.add_error("algorithms", f"Select only {pt} algorithms for a {pt} problem.")
        return data


class LabelForm(forms.Form):
    actual_label = forms.CharField(max_length=255)
