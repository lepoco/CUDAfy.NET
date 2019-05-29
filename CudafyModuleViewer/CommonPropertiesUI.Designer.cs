namespace CudafyModuleViewer
{
    partial class CommonPropertiesUI
    {
        /// <summary> 
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary> 
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Component Designer generated code

        /// <summary> 
        /// Required method for Designer support - do not modify 
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.label5 = new System.Windows.Forms.Label();
            this.label6 = new System.Windows.Forms.Label();
            this.tbName = new System.Windows.Forms.TextBox();
            this.tbType = new System.Windows.Forms.TextBox();
            this.tbIsDummy = new System.Windows.Forms.TextBox();
            this.tbCRCGood = new System.Windows.Forms.TextBox();
            this.tbDeserializedCRC = new System.Windows.Forms.TextBox();
            this.tbAssemblyCRC = new System.Windows.Forms.TextBox();
            this.tbAssembly = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(18, 16);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(38, 13);
            this.label1.TabIndex = 0;
            this.label1.Text = "Name:";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(18, 45);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(34, 13);
            this.label2.TabIndex = 0;
            this.label2.Text = "Type:";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(18, 74);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(120, 13);
            this.label3.TabIndex = 0;
            this.label3.Text = "Deserialized Checksum:";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(18, 152);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(56, 13);
            this.label4.TabIndex = 0;
            this.label4.Text = "Is Dummy:";
            this.label4.Click += new System.EventHandler(this.label4_Click);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(18, 101);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(107, 13);
            this.label5.TabIndex = 0;
            this.label5.Text = "Assembly Checksum:";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(18, 181);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(61, 13);
            this.label6.TabIndex = 0;
            this.label6.Text = "CRC Good:";
            this.label6.Click += new System.EventHandler(this.label4_Click);
            // 
            // tbName
            // 
            this.tbName.Location = new System.Drawing.Point(144, 12);
            this.tbName.Name = "tbName";
            this.tbName.ReadOnly = true;
            this.tbName.Size = new System.Drawing.Size(140, 20);
            this.tbName.TabIndex = 1;
            // 
            // tbType
            // 
            this.tbType.Location = new System.Drawing.Point(144, 42);
            this.tbType.Name = "tbType";
            this.tbType.ReadOnly = true;
            this.tbType.Size = new System.Drawing.Size(140, 20);
            this.tbType.TabIndex = 1;
            // 
            // tbIsDummy
            // 
            this.tbIsDummy.Location = new System.Drawing.Point(144, 149);
            this.tbIsDummy.Name = "tbIsDummy";
            this.tbIsDummy.ReadOnly = true;
            this.tbIsDummy.Size = new System.Drawing.Size(140, 20);
            this.tbIsDummy.TabIndex = 1;
            // 
            // tbCRCGood
            // 
            this.tbCRCGood.Location = new System.Drawing.Point(144, 178);
            this.tbCRCGood.Name = "tbCRCGood";
            this.tbCRCGood.ReadOnly = true;
            this.tbCRCGood.Size = new System.Drawing.Size(140, 20);
            this.tbCRCGood.TabIndex = 1;
            // 
            // tbDeserializedCRC
            // 
            this.tbDeserializedCRC.Location = new System.Drawing.Point(144, 71);
            this.tbDeserializedCRC.Name = "tbDeserializedCRC";
            this.tbDeserializedCRC.ReadOnly = true;
            this.tbDeserializedCRC.Size = new System.Drawing.Size(140, 20);
            this.tbDeserializedCRC.TabIndex = 1;
            // 
            // tbAssemblyCRC
            // 
            this.tbAssemblyCRC.Location = new System.Drawing.Point(144, 97);
            this.tbAssemblyCRC.Name = "tbAssemblyCRC";
            this.tbAssemblyCRC.ReadOnly = true;
            this.tbAssemblyCRC.Size = new System.Drawing.Size(140, 20);
            this.tbAssemblyCRC.TabIndex = 1;
            // 
            // tbAssembly
            // 
            this.tbAssembly.Location = new System.Drawing.Point(144, 123);
            this.tbAssembly.Name = "tbAssembly";
            this.tbAssembly.ReadOnly = true;
            this.tbAssembly.Size = new System.Drawing.Size(140, 20);
            this.tbAssembly.TabIndex = 1;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(18, 126);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(54, 13);
            this.label7.TabIndex = 0;
            this.label7.Text = "Assembly:";
            // 
            // CommonPropertiesUI
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.Controls.Add(this.tbCRCGood);
            this.Controls.Add(this.tbIsDummy);
            this.Controls.Add(this.tbAssembly);
            this.Controls.Add(this.tbAssemblyCRC);
            this.Controls.Add(this.tbDeserializedCRC);
            this.Controls.Add(this.tbType);
            this.Controls.Add(this.tbName);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Name = "CommonPropertiesUI";
            this.Size = new System.Drawing.Size(308, 218);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox tbName;
        private System.Windows.Forms.TextBox tbType;
        private System.Windows.Forms.TextBox tbIsDummy;
        private System.Windows.Forms.TextBox tbCRCGood;
        private System.Windows.Forms.TextBox tbDeserializedCRC;
        private System.Windows.Forms.TextBox tbAssemblyCRC;
        private System.Windows.Forms.TextBox tbAssembly;
        private System.Windows.Forms.Label label7;
    }
}
