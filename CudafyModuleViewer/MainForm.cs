/*
CUDAfy.NET - LGPL 2.1 License
Please consider purchasing a commerical license - it helps development, frees you from LGPL restrictions
and provides you with support.  Thank you!
Copyright (C) 2011 Hybrid DSP Systems
http://www.hybriddsp.com

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Diagnostics;
using Cudafy;
using Cudafy.Compilers;
using Cudafy.Types;
namespace CudafyModuleViewer
{
    public partial class MainForm : Form
    {
        public MainForm(string[] args)
        {
            InitializeComponent();
            Text = Text + string.Format(" ({0}-bit)", IntPtr.Size == 4 ? "32" : "64");

            cbArch.Items.AddRange(Enum.GetNames(typeof(eArchitecture)));
            int index = cbArch.Items.IndexOf(Enum.GetName(typeof(eArchitecture), eArchitecture.sm_20));
            cbArch.SelectedIndex = index;
            try
            {
                if (args.Length > 0)
                {
                    _module = CudafyModule.Deserialize(args[0]);
                    FillForm();
                }
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }       
        }

        private const string csFUNCTIONS_X = "Functions ({0})";
        private const string csCONSTANTS_X = "Constants ({0})";
        private const string csTYPES_X = "Types ({0})";
        private const string csSOURCE_X = "Generated Source Code ({0})";
        private const string csPTX_X = "PTX ({0})";
        private CudafyModule _module;
        private AboutBox _aboutBox;
        private void miAbout_Click(object sender, EventArgs e)
        {
            if (_aboutBox == null)
                _aboutBox = new AboutBox();
            _aboutBox.ShowDialog();
        }

        private void miExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void miOpen_Click(object sender, EventArgs e)
        {
            try
            {
                if (openFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                {
                    _module = CudafyModule.Deserialize(openFileDialog.FileName);
                    FillForm();
                }
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
        }

        private void FillForm()
        {
            lbFunctions.Items.Clear();
            lbConstants.Items.Clear();
            lbTypes.Items.Clear();
            lbPTX.Items.Clear();
            if (_module != null)
            {
                foreach (var item in _module.Functions)
                    lbFunctions.Items.Add(item.Value);
                foreach (var item in _module.Types)
                    lbTypes.Items.Add(item.Value);
                foreach (var item in _module.Constants)
                    lbConstants.Items.Add(item.Value);
                foreach (var item in _module.PTXModules)
                    lbPTX.Items.Add(item);

                lbFunctions.SelectedIndex = _module.Functions.Count > 0 ? 0 : -1;
                lbTypes.SelectedIndex = _module.Types.Count > 0 ? 0 : -1;
                lbConstants.SelectedIndex = _module.Constants.Count > 0 ? 0 : -1;
                lbPTX.SelectedIndex = _module.PTXModules.Length > 0 ? 0 : -1;

                tbSource.Text = _module.SourceCode;
                lbPTX_SelectedIndexChanged(this, null);
                //tbPTX.Text = _module.PTX == null ? string.Empty : _module.PTX.PTX;

                tpConstants.Text = string.Format(csCONSTANTS_X, _module.Constants.Count);
                tpTypes.Text = string.Format(csTYPES_X, _module.Types.Count);
                tpFunctions.Text = string.Format(csFUNCTIONS_X, _module.Functions.Count);
                tpSource.Text = string.Format(csSOURCE_X, _module.HasSourceCode ? "1": "0");
                tpPTX.Text = string.Format(csPTX_X, _module.PTXModules.Length);
            }
            
        }

        private void HandleException(Exception ex)
        {
            MessageBox.Show(ex.Message, "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }

        private void lbFunctions_SelectedIndexChanged(object sender, EventArgs e)
        {
            KernelMethodInfo item = lbFunctions.SelectedItem as KernelMethodInfo;
            if (item != null)
            {
                string text = GetCommon(item as KernelMemberInfo);
                tbFunctions.Text = text;
                text = GetSpecific(item);
                tbFunctions.AppendText(text);
            }
        }

        private void lbTypes_SelectedIndexChanged(object sender, EventArgs e)
        {
            KernelTypeInfo item = lbTypes.SelectedItem as KernelTypeInfo;
            if (item != null)
            {
                string text = GetCommon(item as KernelMemberInfo);
                tbTypes.Text = text;

            }
        }

        private void lbConstants_SelectedIndexChanged(object sender, EventArgs e)
        {
            KernelConstantInfo item = lbConstants.SelectedItem as KernelConstantInfo;
            if (item != null)
            {
                string text = GetCommon(item as KernelMemberInfo);
                tbConstants.Text = text;
                text = GetSpecific(item);
                tbConstants.AppendText(text);
            }
        }

        private void lbPTX_SelectedIndexChanged(object sender, EventArgs e)
        {
            PTXModule item = lbPTX.SelectedItem as PTXModule;
            if (item != null)
            {
                tbPTX.Text = item.PTX;
            }
            else
                tbPTX.Clear();
        }

        private string GetSpecific(KernelMethodInfo kmi)
        {
            StringBuilder sb = new StringBuilder();
            if (kmi != null)
            {
                sb.AppendLine("Parameters            : " + kmi.GetParametersString());
            }
            return sb.ToString();
        }

        private string GetSpecific(KernelConstantInfo kci)
        {
            StringBuilder sb = new StringBuilder();
            if (kci != null)
            {
                sb.AppendLine("Declaration           : " + kci.GetDeclaration());
            }
            return sb.ToString();
        }

        private string GetCommon(KernelMemberInfo kmi)
        {
            StringBuilder sb = new StringBuilder();
            if (kmi != null)
            {
                sb.AppendLine("Name                  : " + kmi.Name);
                sb.AppendLine("Declaring Type        : " + kmi.Type.Name);
                sb.AppendLine("Declaring Assembly    : " + kmi.Type.Assembly.FullName);
                sb.AppendLine("Declaring Assembly CRC: " + kmi.GetAssemblyChecksum().ToString());
                sb.AppendLine("Deserialized CRC      : " + kmi.DeserializedChecksum.ToString());
                sb.AppendLine("Checksum Match?       : " + kmi.TryVerifyChecksums().ToString());
                sb.AppendLine("Is Dummy?             : " + kmi.IsDummy.ToString());
            }
            return sb.ToString();
        }

        private void miEnableEditing_CheckedChanged(object sender, EventArgs e)
        {
            ToolStripMenuItem mi = sender as ToolStripMenuItem;
            if (mi == null)
                return;
            //tbPTX.ReadOnly = !mi.Checked;
            tbSource.ReadOnly = !mi.Checked;
            gbCompile.Enabled = mi.Checked;
        }

        private void btnCompile_Click(object sender, EventArgs e)
        {
            try
            {
                if (_module == null)
                {
                    _module = new CudafyModule();
                }
                
                if (_module != null)
                {
                    eArchitecture arch = (eArchitecture)Enum.Parse(typeof(eArchitecture), cbArch.SelectedItem as string);
                    if(arch == eArchitecture.OpenCL)
                    {
                        MessageBox.Show(this, "OpenCL modules are not compiled.", "Information", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        return;
                    }
                    if (!cb32bit.Checked && !cb64bit.Checked)
                    {
                        MessageBox.Show(this, "Select a platform.", "Information", MessageBoxButtons.OK, MessageBoxIcon.Information);
                        return;
                    }
                    _module.SourceCode = tbSource.Text;
                    
                    NvccCompilerOptions opt = null;
                    _module.CompilerOptionsList.Clear();
                    _module.RemovePTXModules();
                    if (cb64bit.Checked)
                    {
                        opt = NvccCompilerOptions.Createx64(arch);
                        _module.CompilerOptionsList.Add(opt);
                    }
                    if (cb32bit.Checked)
                    {
                        opt = NvccCompilerOptions.Createx86(arch);
                        _module.CompilerOptionsList.Add(opt);
                    }

                    
                    _module.Compile(eGPUCompiler.CudaNvcc, false);
                    FillForm();
                }
            }
            catch (Exception ex)
            {
                HandleException(ex);
            }
        }

        private void miSaveAs_Click(object sender, EventArgs e)
        {
            
            if (_module != null && saveFileDialog.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                try
                {
                    _module.SourceCode = tbSource.Text;                   
                    _module.Serialize(saveFileDialog.FileName);
                }
                catch (Exception ex)
                {
                    HandleException(ex);
                }
            }
        }

        private void miFile_DropDownOpening(object sender, EventArgs e)
        {
            miSaveAs.Enabled = _module != null && miEnableEditing.Checked;
        }

        private void btnInstallGAC_Click(object sender, EventArgs e)
        {
            OpenFile("http://support.microsoft.com/kb/815808");
        }

        private bool OpenFile(string filename, string args = null)
        {
            try
            {
                Process process = new Process();
                process.StartInfo.FileName = filename;
                if (args != null)
                    process.StartInfo.Arguments = args;
                process.Start();
            }
            catch (Exception ex)
            {
                Trace.WriteLine(ex.Message);
                return false;
            }
            return true;
        }

        private void btnCUDACheck_Click(object sender, EventArgs e)
        {
            tbCUDA.Clear();
            foreach (string s in CUDACheck.EnumerateDevices())
                tbCUDA.AppendText(s + "\r\n");
        }

        private void btnResolveCUDADeviceIssue_Click(object sender, EventArgs e)
        {
            OpenFile("http://developer.nvidia.com/cuda");
        }

        private void btnTestCUDA_Click(object sender, EventArgs e)
        {
            tbCUDA.Clear();
            try
            {
                foreach (string s in CUDACheck.TestCUDASDK())
                    tbCUDA.AppendText(s + "\r\n");
            }
            catch (Exception ex)
            {
                tbCUDA.AppendText(ex.Message + "\r\n");
            }
        }

        private void btnResolveCUDAIssue_Click(object sender, EventArgs e)
        {
            OpenFile("http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/index.html");
        }

        private void btnCheckOpenCL_Click(object sender, EventArgs e)
        {

        }

        private void btnCheckOpenCL_Click_1(object sender, EventArgs e)
        {
            tbOpenCL.Clear();
            foreach (string s in CUDACheck.EnumerateDevices(true))
                tbOpenCL.AppendText(s + "\r\n");
        }

        private void btnTestOpenCL_Click(object sender, EventArgs e)
        {
            tbOpenCL.Clear();
            try
            {
                foreach (string s in CUDACheck.TestOpenCL())
                    tbOpenCL.AppendText(s + "\r\n");
            }
            catch (Exception ex)
            {
                tbOpenCL.AppendText(ex.Message + "\r\n");
            }
        }

        private void btnVisitAMDOpenCL_Click(object sender, EventArgs e)
        {
            OpenFile("http://developer.amd.com/tools/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/");
        }

        private void btnVisitIntelOpenCL_Click(object sender, EventArgs e)
        {
            OpenFile("http://software.intel.com/en-us/vcsource/tools/opencl-sdk");
        }

        private void tsmiVisitForum_Click(object sender, EventArgs e)
        {
            OpenFile("https://cudafy.codeplex.com/discussions");
        }

        private void tsmiVisitDoc_Click(object sender, EventArgs e)
        {
            OpenFile("http://www.hybriddsp.com/Products/CUDAfyNET.aspx");
        }

        private void tsmiDoc_Click(object sender, EventArgs e)
        {
            OpenFile("https://cudafy.codeplex.com/documentation");
        }
    }
}   
